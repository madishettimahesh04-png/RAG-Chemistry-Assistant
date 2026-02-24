import os
import gdown
import zipfile
import streamlit as st
import pandas as pd
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from groq import Groq

st.title("🧪LLM MnSol ΔG Solvation Assistant")

# -------------------------
# DOWNLOAD FAISS
# -------------------------
FILE_ID = "1gUKTTKNjOqI2jP3I6bGVSmFtb3JD7jn2"
ZIP_FILE = "mnsol_faiss_index.zip"

if not os.path.exists("mnsol_faiss_index"):

    with st.spinner("Downloading vector database..."):

        url = f"https://drive.google.com/uc?id={FILE_ID}"
        gdown.download(url, ZIP_FILE, quiet=False)

        with zipfile.ZipFile(ZIP_FILE, 'r') as zip_ref:
            zip_ref.extractall(".")


# -------------------------
# LOAD DATA
# -------------------------
df = pd.read_csv("new_dataset_with_predictions.csv")


# -------------------------
# LOAD VECTORSTORE
# -------------------------
@st.cache_resource
def load_vs():

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    return FAISS.load_local(
        "mnsol_faiss_index",
        embeddings,
        allow_dangerous_deserialization=True
    )

vectorstore = load_vs()
retriever = vectorstore.as_retriever(search_kwargs={"k":3})


# -------------------------
# GROQ API
# -------------------------
client = Groq(api_key=st.secrets["GROQ_API_KEY"])


query = st.text_input("Ask question")

if query:
    docs = retriever.invoke(query)
    context = "\n\n".join([d.page_content for d in docs])

    completion = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
{
"role":"system",
"content":"""
You are a chemistry assistant specialized in solvation free energy prediction.

Instructions:
- Extract the ΔG solvation value from the provided dataset context.
- Output ONLY:
    1. DeltaG value
    2. Short scientific explanation (1–2 sentences).
- Do NOT add greetings or extra text.
- Do NOT invent values.
- If value is missing, say: DeltaG not found in dataset.
"""
},
{
"role":"user",
"content":f"""
Dataset Context:
{context}

Question:
{query}
"""
}
]
    )

    st.write(completion.choices[0].message.content)
