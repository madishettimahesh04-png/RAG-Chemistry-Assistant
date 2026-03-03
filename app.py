# ==========================================================
# 🤖 MnSol Conversational ΔG Assistant
# ==========================================================

import os
import re
import gdown
import zipfile
import streamlit as st
import pandas as pd

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import BM25Retriever
from groq import Groq
from rapidfuzz import process, fuzz


# ------------------------------------------------
# PAGE CONFIG
# ------------------------------------------------
st.set_page_config(
    page_title="MnSol ΔG Assistant",
    page_icon="🧪",
    layout="centered"
)

# ------------------------------------------------
# STYLING
# ------------------------------------------------
st.markdown("""
<style>
[data-testid="stChatMessage"] {
    border-radius: 18px;
    padding: 15px;
    margin-bottom: 10px;
}
[data-testid="stChatMessageContent"] {
    font-size: 16px;
}
.stChatInput textarea {
    border-radius: 12px;
}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<h1 style="
text-align:center;
background: linear-gradient(90deg,#00c6ff,#0072ff);
-webkit-background-clip: text;
color: transparent;">
🧪 MnSol AI Chemistry Assistant
</h1>
""", unsafe_allow_html=True)

st.caption("Ask anything about solvation free energy (ΔGsolv)")


# ------------------------------------------------
# DOWNLOAD VECTOR DB
# ------------------------------------------------
FILE_ID = "1gUKTTKNjOqI2jP3I6bGVSmFtb3JD7jn2"
ZIP_FILE = "mnsol_faiss_index.zip"
INDEX_FOLDER = "mnsol_faiss_index"

if not os.path.exists(INDEX_FOLDER):
    with st.spinner("Downloading vector database..."):
        url = f"https://drive.google.com/uc?id={FILE_ID}"
        gdown.download(url, ZIP_FILE, quiet=False)
        with zipfile.ZipFile(ZIP_FILE, "r") as zip_ref:
            zip_ref.extractall(".")


# ------------------------------------------------
# LOAD DATASET
# ------------------------------------------------
df = pd.read_csv("five_columns_dataset_with_predictions_3.csv")

solute_list = df["SoluteName"].astype(str).unique().tolist()
solvent_list = df["Solvent"].astype(str).unique().tolist()

formula_to_solute = dict(
    zip(df["Formula"].astype(str).str.lower(), df["SoluteName"])
)


# ------------------------------------------------
# MATCHING FUNCTIONS
# ------------------------------------------------
def resolve_solute(user_input):
    key = user_input.lower().strip()
    if key in formula_to_solute:
        return formula_to_solute[key]

    match, score, _ = process.extractOne(
        user_input,
        solute_list,
        scorer=fuzz.WRatio
    )
    return match if score > 75 else None


def match_solvent(user_input):
    match, score, _ = process.extractOne(
        user_input,
        solvent_list,
        scorer=fuzz.WRatio
    )
    return match if score > 75 else None


def try_exact_from_query(text):
    """
    Attempt to auto-detect solute, solvent, charge
    from natural language question.
    """

    found_solute = None
    found_solvent = None
    found_charge = None

    # Detect charge like +1, -1, 2+
    charge_match = re.search(r"([+-]?\d+)", text)
    if charge_match:
        try:
            found_charge = int(charge_match.group())
        except:
            pass

    # Detect solute & solvent via fuzzy match
    for word in text.split():
        sol = resolve_solute(word)
        solv = match_solvent(word)

        if sol:
            found_solute = sol
        if solv:
            found_solvent = solv

    if found_solute and found_solvent:
        result = df[
            (df["SoluteName"] == found_solute) &
            (df["Solvent"] == found_solvent)
        ]

        if found_charge is not None:
            result = result[result["Charge"] == found_charge]

        if len(result) > 0:
            return result.iloc[0]["Predicted_DeltaGsolv"], found_solute, found_solvent

    return None, None, None


# ------------------------------------------------
# VECTOR STORE
# ------------------------------------------------
@st.cache_resource
def load_vectorstore():
    embeddings = HuggingFaceEmbeddings(
        model_name="BAAI/bge-base-en-v1.5"
    )
    return FAISS.load_local(
        INDEX_FOLDER,
        embeddings,
        allow_dangerous_deserialization=True
    )


vectorstore = load_vectorstore()

docs = list(vectorstore.docstore._dict.values())
bm25 = BM25Retriever.from_documents(docs)
bm25.k = 3

vector_retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 3}
)


def hybrid_search(query):
    semantic = vector_retriever.invoke(query)
    keyword = bm25.invoke(query)

    combined = {
        doc.page_content: doc
        for doc in semantic + keyword
    }

    return list(combined.values())[:5]


# ------------------------------------------------
# GROQ
# ------------------------------------------------
client = Groq(api_key=st.secrets["GROQ_API_KEY"])


# ------------------------------------------------
# CHAT MEMORY
# ------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []


for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])


# ------------------------------------------------
# CHAT INPUT
# ------------------------------------------------
prompt = st.chat_input("Ask about ΔGsolv, molecules, solvents...")

if prompt:

    st.session_state.messages.append(
        {"role": "user", "content": prompt}
    )

    with st.chat_message("user"):
        st.markdown(prompt)

    # 1️⃣ Try exact auto-detection
    deltag, solute, solvent = try_exact_from_query(prompt)

    if deltag is not None:

        response = f"""
The predicted solvation free energy (ΔGsolv) of **{solute}**
in **{solvent}** is **{deltag} kcal/mol**.

A negative value indicates thermodynamically favorable
solute–solvent interactions, typically driven by
ion–dipole or hydrogen bonding effects.

A positive value would suggest weak or unfavorable
interactions relative to the gas phase.
"""

    else:

        # 2️⃣ Hybrid RAG fallback
        docs = hybrid_search(prompt)
        context = "\n\n".join(
            [doc.page_content for doc in docs]
        )

        completion = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {
                    "role": "system",
                    "content": """You are an expert physical chemist.
Answer scientifically.
Explain molecular interactions clearly.
If data exists in context, use it."""
                },
                {
                    "role": "user",
                    "content": context + "\n\nQuestion: " + prompt
                }
            ]
        )

        response = completion.choices[0].message.content

    with st.chat_message("assistant"):
        st.markdown(response)

    st.session_state.messages.append(
        {"role": "assistant", "content": response}
    )
