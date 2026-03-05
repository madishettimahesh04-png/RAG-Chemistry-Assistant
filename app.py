import os
import re
import zipfile
import gdown
import streamlit as st
import pandas as pd
from rapidfuzz import process, fuzz
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from groq import Groq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.output_parsers import StrOutputParser

# ────────────────────────────────────────────────
#  Constants & Config
# ────────────────────────────────────────────────
FILE_ID = "1gUKTTKNjOqI2jP3I6bGVSmFtb3JD7jn2"
ZIP_FILE = "mnsol_faiss_index.zip"
INDEX_FOLDER = "mnsol_faiss_index"
DATA_CSV = "five_columns_dataset_with_predictions_3.csv"

st.set_page_config(page_title="MnSol ΔG Assistant", layout="wide")

# Modern gradient title
st.markdown("""
<h1 style='text-align:center; background: linear-gradient(90deg, #00c6ff, #0072ff); -webkit-background-clip: text; color: transparent;'>
    🧪 MnSol AI Solvation Free Energy Assistant
</h1>
""", unsafe_allow_html=True)

# ────────────────────────────────────────────────
#  Cached heavy resources   ←  very important
# ────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading embeddings model...")
def load_embeddings():
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

@st.cache_resource(show_spinner="Loading FAISS index (may take ~20–40s first time)...")
def load_faiss_index():
    if not os.path.exists(INDEX_FOLDER):
        with st.spinner("Downloading & extracting FAISS index from Google Drive..."):
            url = f"https://drive.google.com/uc?id={FILE_ID}"
            gdown.download(url, ZIP_FILE, quiet=True)
            with zipfile.ZipFile(ZIP_FILE, "r") as zip_ref:
                zip_ref.extractall(".")
    return FAISS.load_local(INDEX_FOLDER, embeddings=load_embeddings(), allow_dangerous_deserialization=True)

@st.cache_resource
def get_groq_client():
    return Groq(api_key=st.secrets["GROQ_API_KEY"])

@st.cache_data
def load_dataset():
    return pd.read_csv(DATA_CSV)

# Load once
embeddings = load_embeddings()
vectorstore = load_faiss_index()
groq_client = get_groq_client()
df = load_dataset()

solute_list = df["SoluteName"].astype(str).unique().tolist()
solvent_list = df["Solvent"].astype(str).unique().tolist()

# ────────────────────────────────────────────────
#  Fuzzy matching (unchanged but can be cached if needed)
# ────────────────────────────────────────────────
def resolve_solute(name: str) -> str | None:
    match, score, _ = process.extractOne(name, solute_list, scorer=fuzz.WRatio)
    return match if score > 78 else None   # slightly stricter

def resolve_solvent(name: str) -> str | None:
    match, score, _ = process.extractOne(name, solvent_list, scorer=fuzz.WRatio)
    return match if score > 78 else None

# ────────────────────────────────────────────────
#  Better query parser → tries exact match first, then RAG
# ────────────────────────────────────────────────
def extract_solute_solvent_charge(text: str):
    text = text.lower()
    charge = None
    m = re.search(r"([+-]?\d+)[ ]*(?:charge|e?|z)?", text)
    if m:
        try:
            charge = int(m.group(1))
        except:
            pass

    words = re.findall(r"\w+", text)
    solute, solvent = None, None
    for w in words:
        if not solute:
            solute = resolve_solute(w)
        if not solvent:
            solvent = resolve_solvent(w)
        if solute and solvent:
            break
    return solute, solvent, charge

# ────────────────────────────────────────────────
#  Main response logic
# ────────────────────────────────────────────────
def generate_response(user_msg: str, history: list):
    solute, solvent, charge = extract_solute_solvent_charge(user_msg)

    # ── Fast path: exact table lookup ──
    if solute and solvent:
        query = (df["SoluteName"] == solute) & (df["Solvent"] == solvent)
        if charge is not None:
            query &= (df["Charge"] == charge)
        hit = df[query]
        if not hit.empty:
            row = hit.iloc[0]
            dg = row["Predicted_DeltaGsolv"]
            txt = f"**Exact match found!**\n\n"
            txt += f"**{solute}** in **{solvent}** → **ΔG_solv = {dg:.2f} kcal/mol**\n"
            if charge is not None:
                txt += f"(charge = {charge})\n"
            txt += "\n**Quick thermodynamic interpretation:**\n"
            # You can cache or pre-compute explanations for top solutes if you want
            expl = groq_client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[
                    {"role": "system", "content": "Give exactly 3 concise scientific sentences explaining ΔGsolv value and main interactions."},
                    {"role": "user", "content": f"Solute: {solute}, Solvent: {solvent}, ΔG = {dg} kcal/mol"}
                ],
                temperature=0.4,
                max_tokens=180
            ).choices[0].message.content
            return txt + expl, True   # True = from database

    # ── RAG fallback ──
    docs = vectorstore.similarity_search(user_msg, k=5)
    context = "\n\n".join([d.page_content for d in docs])

    system_prompt = """You are an expert computational chemist specializing in solvation free energies.
Use the following context from the MnSol database and scientific literature when relevant.
Be precise, scientific, and concise. If unsure — say so."""

    messages = [
        {"role": "system", "content": system_prompt + "\n\nContext:\n" + context},
        *history[-12:],   # last 6 turns ≈ 12 messages
        {"role": "user", "content": user_msg}
    ]

    stream = groq_client.chat.completions.create(
        model="llama-3.1-70b-versatile",   # or mixtral-8x22b-2404 if allowed
        messages=messages,
        temperature=0.35,
        max_tokens=600,
        stream=True
    )

    return stream, False

# ────────────────────────────────────────────────
#  Session state & chat history
# ────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "👋 Hi! I'm your MnSol ΔG assistant.\nAsk me about solvation free energies!\n\nExamples:\n• benzene in water\n• Na+ in methanol\n• What is ΔGsolv of ethanol in acetone?"}
    ]

# Show history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ────────────────────────────────────────────────
#  Chat input + streaming
# ────────────────────────────────────────────────
if prompt := st.chat_input("Ask about ΔG_solv (solute + solvent or general question)"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            stream_or_text, from_db = generate_response(prompt, st.session_state.messages)

        if from_db:
            st.markdown(stream_or_text)
            full_response = stream_or_text
        else:
            full_response = ""
            placeholder = st.empty()
            for chunk in stream_or_text:
                if token := chunk.choices[0].delta.content:
                    full_response += token
                    placeholder.markdown(full_response + "▌")
            placeholder.markdown(full_response)

    st.session_state.messages.append({"role": "assistant", "content": full_response})

