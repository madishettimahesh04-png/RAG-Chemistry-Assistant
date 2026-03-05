# ==========================================================
# MnSol ΔG Assistant – Professional Edition 2026
# ==========================================================

import os
import re
import zipfile
import gdown
import streamlit as st
import pandas as pd
import plotly.express as px
from rapidfuzz import process, fuzz
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from groq import Groq
from datetime import datetime


# ----------------------------------------------------------
# System Prompt
# ----------------------------------------------------------

SYSTEM_PROMPT = """
You are MnSol-AI, an expert assistant in computational solvation chemistry.

You specialize in:
• Solvation free energy (ΔG_solv)
• Molecular interactions in solvents
• Thermodynamic interpretation
• The MnSol dataset

Provide concise scientific answers suitable for researchers.
Prefer quantitative reasoning when possible.
"""


# ----------------------------------------------------------
# Page Config
# ----------------------------------------------------------

st.set_page_config(
    page_title="MnSol ΔG Assistant",
    page_icon="🧪",
    layout="wide"
)

# ----------------------------------------------------------
# CSS Styling
# ----------------------------------------------------------

st.markdown("""
<style>
.title-gradient {
background: linear-gradient(90deg,#00c6ff,#0072ff);
-webkit-background-clip:text;
-webkit-text-fill-color:transparent;
font-size:2.6rem;
font-weight:700;
text-align:center;
margin-bottom:1rem;
}

[data-testid="stChatMessage"]{
border-radius:14px;
padding:14px;
margin-bottom:8px;
}

.example-box{
background:#f8f9fa;
padding:14px;
border-radius:10px;
border:1px solid #e0e0e0;
}
</style>
""", unsafe_allow_html=True)


# ----------------------------------------------------------
# Cached Resources
# ----------------------------------------------------------

@st.cache_resource
def get_embeddings():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )


@st.cache_resource
def load_vectorstore():

    INDEX_FOLDER = "mnsol_faiss_index"

    if not os.path.exists(INDEX_FOLDER):

        with st.spinner("Downloading vector database..."):

            url = "https://drive.google.com/uc?id=1gUKTTKNjOqI2jP3I6bGVSmFtb3JD7jn2"

            gdown.download(url, "mnsol_faiss_index.zip", quiet=False)

            with zipfile.ZipFile("mnsol_faiss_index.zip","r") as z:
                z.extractall(".")

    return FAISS.load_local(
        INDEX_FOLDER,
        embeddings=get_embeddings(),
        allow_dangerous_deserialization=True
    )


@st.cache_resource
def get_groq():
    return Groq(api_key=st.secrets["GROQ_API_KEY"])


@st.cache_data
def load_data():
    return pd.read_csv("five_columns_dataset_with_predictions_3.csv")


vector_db = load_vectorstore()
groq = get_groq()
df = load_data()

solute_list = df["SoluteName"].astype(str).unique().tolist()
solvent_list = df["Solvent"].astype(str).unique().tolist()


# ----------------------------------------------------------
# Fuzzy Matching
# ----------------------------------------------------------

def find_solute(text):

    match, score, _ = process.extractOne(
        text, solute_list, scorer=fuzz.WRatio
    )

    return match if score > 78 else None


def find_solvent(text):

    match, score, _ = process.extractOne(
        text, solvent_list, scorer=fuzz.WRatio
    )

    return match if score > 78 else None


# ----------------------------------------------------------
# Query Parsing
# ----------------------------------------------------------

def parse_query(text):

    text = text.lower()
    text = text.replace("/", " ").replace("→"," ")

    charge = None

    charge_match = re.search(r"([+-]?\d+)", text)

    if charge_match:
        try:
            charge = int(charge_match.group())
        except:
            pass

    words = re.findall(r"[a-zA-Z0-9+\-().]+", text)

    solute = None
    solvent = None

    for w in words:

        if not solute:
            s = find_solute(w)
            if s:
                solute = s

        if not solvent:
            s = find_solvent(w)
            if s:
                solvent = s

    return solute, solvent, charge


# ----------------------------------------------------------
# Response Generator
# ----------------------------------------------------------

def generate_answer(question, history):

    solute, solvent, charge = parse_query(question)

    if solute and solvent:

        mask = (df["SoluteName"]==solute) & (df["Solvent"]==solvent)

        if charge is not None:
            mask &= (df["Charge"]==charge)

        result = df[mask]

        if not result.empty:

            row = result.iloc[0]
            deltag = row["Predicted_DeltaGsolv"]

            base = f"""
**Exact MnSol match**

Solute: **{solute}**  
Solvent: **{solvent}**

**ΔG_solv = {deltag:.2f} kcal/mol**
"""

            explanation = groq.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[
                    {"role":"system","content":SYSTEM_PROMPT},
                    {"role":"user",
                    "content":f"Explain thermodynamic meaning of ΔGsolv {deltag} kcal/mol for {solute} in {solvent}."}
                ],
                temperature=0.3,
                max_tokens=200
            )

            return base + "\n\n" + explanation.choices[0].message.content, True, None


    docs = vector_db.similarity_search(question, k=4)

    context = "\n\n".join(d.page_content for d in docs)

    messages = [
        {"role":"system","content":SYSTEM_PROMPT + "\n\nContext:\n"+context},
        *history[-12:],
        {"role":"user","content":question}
    ]

    stream = groq.chat.completions.create(
        model="llama-3.1-70b-versatile",
        messages=messages,
        temperature=0.4,
        max_tokens=600,
        stream=True
    )

    return stream, False, None


# ----------------------------------------------------------
# Session State
# ----------------------------------------------------------

if "messages" not in st.session_state:
    st.session_state.messages=[]


# ----------------------------------------------------------
# Tabs
# ----------------------------------------------------------

tab_chat, tab_about, tab_data = st.tabs(
    ["Chat","About","Dataset"]
)


# ----------------------------------------------------------
# Chat
# ----------------------------------------------------------

with tab_chat:

    if not st.session_state.messages:

        st.markdown('<div class="title-gradient">MnSol ΔG Solvation Assistant</div>', unsafe_allow_html=True)

        st.markdown("""
<div class="example-box">

Example queries

benzene in water  
Na+ in methanol charge +1  
Why hydrocarbons have positive ΔG in water?

</div>
""", unsafe_allow_html=True)


    for msg in st.session_state.messages:

        avatar = "🧑‍🔬" if msg["role"]=="user" else "🧪"

        with st.chat_message(msg["role"], avatar=avatar):
            st.markdown(msg["content"])


    if prompt := st.chat_input("Ask about solvation free energy..."):

        st.session_state.messages.append({"role":"user","content":prompt})

        with st.chat_message("assistant", avatar="🧪"):

            result, exact, _ = generate_answer(prompt, st.session_state.messages)

            placeholder = st.empty()
            full=""

            if exact:

                full=result
                placeholder.markdown(full)

            else:

                for chunk in result:

                    if delta:=chunk.choices[0].delta.content:

                        full+=delta
                        placeholder.markdown(full+"▌")

                placeholder.markdown(full)

        st.session_state.messages.append(
            {"role":"assistant","content":full}
        )


# ----------------------------------------------------------
# About
# ----------------------------------------------------------

with tab_about:

    st.header("About MnSol Assistant")

    st.markdown("""
AI assistant for exploring **solvation free energy (ΔG_solv)** from the **MnSol dataset**.

Technology

• Streamlit  
• FAISS semantic search  
• HuggingFace embeddings  
• Groq Llama-3.1  
• RapidFuzz matching
""")


# ----------------------------------------------------------
# Dataset
# ----------------------------------------------------------

with tab_data:

    st.metric("Unique Solutes", len(solute_list))
    st.metric("Unique Solvents", len(solvent_list))
    st.metric("Total Systems", len(df))

    fig = px.histogram(
        df,
        x="Predicted_DeltaGsolv",
        nbins=40,
        title="Distribution of ΔG_solv"
    )

    st.plotly_chart(fig, use_container_width=True)

    st.dataframe(
        df.head(300)[["SoluteName","Solvent","Charge","Predicted_DeltaGsolv"]],
        use_container_width=True
    )
