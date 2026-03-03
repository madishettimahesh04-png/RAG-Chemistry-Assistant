# ==========================================================
# 🤖 AI-Powered MnSol ΔG Assistant (ChatGPT Style Version)
# ==========================================================

import os
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
# MODERN STYLING
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
h1 {
    text-align: center;
}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<h1 style="
background: linear-gradient(90deg,#00c6ff,#0072ff);
-webkit-background-clip: text;
color: transparent;">
🧪 AI-Powered MnSol ΔG Assistant
</h1>
""", unsafe_allow_html=True)

st.caption("Solvation Free Energy Assistant")

# ------------------------------------------------
# DOWNLOAD FAISS INDEX
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
    zip(
        df["Formula"].astype(str).str.lower(),
        df["SoluteName"]
    )
)


# ------------------------------------------------
# MATCH FUNCTIONS
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
    return match if score > 75 else user_input


def match_solvent(user_input):
    match, score, _ = process.extractOne(
        user_input,
        solvent_list,
        scorer=fuzz.WRatio
    )
    return match if score > 75 else user_input


def exact_lookup(solute, solvent, charge):
    result = df[
        (df["SoluteName"].str.lower() == solute.lower()) &
        (df["Solvent"].str.lower() == solvent.lower())
    ]

    if charge != "":
        try:
            charge_int = int(charge)
            result = result[result["Charge"] == charge_int]
        except:
            pass

    if len(result) > 0:
        return result.iloc[0]["Predicted_DeltaGsolv"]

    return None


# ------------------------------------------------
# LOAD VECTOR STORE
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


def create_retrievers(vectorstore):
    vector_retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 3}
    )

    docs = list(vectorstore.docstore._dict.values())
    bm25 = BM25Retriever.from_documents(docs)
    bm25.k = 3

    return vector_retriever, bm25


vector_retriever, bm25_retriever = create_retrievers(vectorstore)


def hybrid_search(query):
    semantic = vector_retriever.invoke(query)
    keyword = bm25_retriever.invoke(query)

    combined = {
        doc.page_content: doc
        for doc in semantic + keyword
    }

    return list(combined.values())[:5]


# ------------------------------------------------
# GROQ CLIENT
# ------------------------------------------------
client = Groq(api_key=st.secrets["GROQ_API_KEY"])


# ------------------------------------------------
# CHAT MEMORY
# ------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display previous messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])


# ------------------------------------------------
# CHAT INPUT
# ------------------------------------------------
prompt = st.chat_input(
    "Enter: Solute, Charge, Solvent  (Example: Na+, 1, water)"
)

if prompt:

    st.session_state.messages.append(
        {"role": "user", "content": prompt}
    )

    with st.chat_message("user"):
        st.markdown(prompt)

    # Parse input
    try:
        formula, charge, solvent = [
            x.strip() for x in prompt.split(",")
        ]
    except:
        response = "⚠️ Format: Solute, Charge, Solvent"
        st.session_state.messages.append(
            {"role": "assistant", "content": response}
        )
        with st.chat_message("assistant"):
            st.markdown(response)
        st.stop()

    matched_solute = resolve_solute(formula)
    matched_solvent = match_solvent(solvent)

    deltag = exact_lookup(
        matched_solute,
        matched_solvent,
        charge
    )

    # ------------------------------------------------
    # EXACT MATCH
    # ------------------------------------------------
    if deltag is not None:

        response = f"""
### ✅ Exact Dataset Match

**Solute:** {matched_solute}  
**Solvent:** {matched_solvent}  
**Charge:** {charge}  

### 💧 ΔGₛₒₗᵥ = {deltag} kcal/mol

---

### 🔬 What is happening?

Solvation free energy (ΔGsolv) measures the energy change 
when the molecule moves from gas phase to solvent.

• Negative ΔG → Favorable solvation  
• Positive ΔG → Unfavorable solvation  

This value is retrieved directly from MnSol dataset.
"""

    # ------------------------------------------------
    # HYBRID RAG
    # ------------------------------------------------
    else:

        query = f"""
        SoluteName {matched_solute}
        Solvent {matched_solvent}
        Charge {charge}
        solvation free energy DeltaG
        """

        docs = hybrid_search(query)
        context = "\n\n".join(
            [doc.page_content for doc in docs]
        )

        completion = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {
                    "role": "system",
                    "content": """You are a chemistry expert.
Explain:
1. DeltaG value
2. Molecular interactions
3. Why positive/negative
Use dataset context only."""
                },
                {
                    "role": "user",
                    "content": context
                }
            ]
        )

        response = completion.choices[0].message.content

    with st.chat_message("assistant"):
        st.markdown(response)

    st.session_state.messages.append(
        {"role": "assistant", "content": response}
    )
