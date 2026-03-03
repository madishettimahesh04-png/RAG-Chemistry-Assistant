# ==========================================================
# 🤖 MnSol Conversational ΔG Assistant (Refined Version)
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
from langchain_core.documents import Document
from groq import Groq
from rapidfuzz import process, fuzz
from typing import Optional, Tuple

# ------------------------------------------------
# UI CONFIG
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
🧪 MnSol AI Solvation Assistant
</h1>
""", unsafe_allow_html=True)

# ==========================================================
# INTENT DETECTION (Refined with more intents and robustness)
# ==========================================================
def is_greeting(text: str) -> bool:
    greetings = [
        r"\b(hi|hello|hey|hi there)\b",
        r"\b(good morning|good afternoon|good evening|greetings|howdy)\b"
    ]
    text_lower = text.lower()
    return any(re.search(pattern, text_lower) for pattern in greetings)

def is_small_talk(text: str) -> bool:
    small_talk = [
        r"\b(how are you|how are you doing|how's it going|what's up)\b",
        r"\b(fine|great|good|okay)\b.*\b(and you)\b"
    ]
    text_lower = text.lower()
    return any(re.search(pattern, text_lower) for pattern in small_talk)

def is_identity_question(text: str) -> bool:
    identity_phrases = [
        r"\b(who are you|what is your name|tell me about yourself|introduce yourself|what are you)\b",
        r"\b(your name)\b"
    ]
    text_lower = text.lower()
    return any(re.search(pattern, text_lower) for pattern in identity_phrases)

def is_thanks(text: str) -> bool:
    thanks_words = [
        r"\b(thank you|thanks|thx|cheers)\b"
    ]
    text_lower = text.lower()
    return any(re.search(pattern, text_lower) for pattern in thanks_words)

def is_goodbye(text: str) -> bool:
    goodbye_words = [
        r"\b(bye|goodbye|see you|farewell|talk later)\b"
    ]
    text_lower = text.lower()
    return any(re.search(pattern, text_lower) for pattern in goodbye_words)

def is_help_request(text: str) -> bool:
    help_phrases = [
        r"\b(help|how to|guide|tutorial|explain|what is)\b.*\b(ΔG|solvation|solvent|solute)\b",
        r"\b(how does it work|usage|examples)\b"
    ]
    text_lower = text.lower()
    return any(re.search(pattern, text_lower) for pattern in help_phrases)

def is_irrelevant(text: str) -> bool:
    chemistry_keywords = [
        "solvent", "solute", "delta", "deltag", "ΔG", "solvation",
        "water", "methanol", "ethanol", "ion", "molecule", "benzene",
        "kcal", "mol", "charge", "thermodynamic"
    ]
    text_lower = text.lower()
    has_chem = any(re.search(r"\b" + re.escape(word) + r"\b", text_lower) for word in chemistry_keywords)
    return not has_chem

def detect_intent(text: str) -> str:
    """Detect primary intent with priority order."""
    if is_greeting(text):
        return "greeting"
    elif is_small_talk(text):
        return "small_talk"
    elif is_identity_question(text):
        return "identity"
    elif is_help_request(text):
        return "help"
    elif is_thanks(text):
        return "thanks"
    elif is_goodbye(text):
        return "goodbye"
    elif is_irrelevant(text):
        return "irrelevant"
    else:
        return "query"

# ------------------------------------------------
# IDENTITY DETECTION (Integrated into intent)
# ------------------------------------------------

# ==========================================================
# LOAD DATA (Refined with vector store integration)
# ==========================================================
FILE_ID = "1gUKTTKNjOqI2jP3I6bGVSmFtb3JD7jn2"
ZIP_FILE = "mnsol_faiss_index.zip"
INDEX_FOLDER = "mnsol_faiss_index"

@st.cache_resource
def load_data():
    if not os.path.exists(INDEX_FOLDER):
        with st.spinner("Downloading vector database..."):
            url = f"https://drive.google.com/uc?id={FILE_ID}"
            gdown.download(url, ZIP_FILE, quiet=False)
            with zipfile.ZipFile(ZIP_FILE, "r") as zip_ref:
                zip_ref.extractall(".")
    
    df = pd.read_csv("five_columns_dataset_with_predictions_3.csv")
    solute_list = df["SoluteName"].astype(str).unique().tolist()
    solvent_list = df["Solvent"].astype(str).unique().tolist()
    
    # Prepare documents for FAISS
    documents = []
    for _, row in df.iterrows():
        content = f"Solute: {row['SoluteName']}, Solvent: {row['Solvent']}, Charge: {row['Charge']}, Predicted ΔGsolv: {row['Predicted_DeltaGsolv']} kcal/mol"
        documents.append(Document(page_content=content, metadata={"deltag": row['Predicted_DeltaGsolv'], "solute": row['SoluteName'], "solvent": row['Solvent'], "charge": row['Charge']}))
    
    # Load or create FAISS vector store
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    if os.path.exists(os.path.join(INDEX_FOLDER, "index.faiss")):
        vectorstore = FAISS.load_local(INDEX_FOLDER, embeddings, allow_dangerous_deserialization=True)
    else:
        vectorstore = FAISS.from_documents(documents, embeddings)
        vectorstore.save_local(INDEX_FOLDER)
    
    # BM25 retriever for keyword fallback
    texts = [doc.page_content for doc in documents]
    bm25_retriever = BM25Retriever.from_texts(texts, k=3)
    
    return df, solute_list, solvent_list, vectorstore, bm25_retriever

df, solute_list, solvent_list, vectorstore, bm25_retriever = load_data()

# ==========================================================
# MATCHING FUNCTIONS (Refined with better parsing and fallback)
# ==========================================================
def extract_entities(text: str) -> Tuple[Optional[str], Optional[str], Optional[int]]:
    """Extract solute, solvent, charge using regex and fuzzy matching."""
    text_lower = text.lower()
    
    # Charge extraction
    charge_match = re.search(r"([+-]?\d+)", text)
    charge = int(charge_match.group()) if charge_match else None
    
    # Split and match
    candidates = re.findall(r'\b\w+\b', text_lower)
    solute = None
    solvent = None
    
    for candidate in candidates:
        sol_match = resolve_solute(candidate)
        solv_match = match_solvent(candidate)
        if sol_match and not solute:
            solute = sol_match
        if solv_match and not solvent:
            solvent = solv_match
    
    return solute, solvent, charge

def resolve_solute(user_input: str) -> Optional[str]:
    match, score, _ = process.extractOne(user_input, solute_list, scorer=fuzz.WRatio)
    return match if score > 75 else None

def match_solvent(user_input: str) -> Optional[str]:
    match, score, _ = process.extractOne(user_input, solvent_list, scorer=fuzz.WRatio)
    return match if score > 75 else None

def try_exact_match(solute: str, solvent: str, charge: Optional[int]) -> Optional[Tuple[float, str, str]]:
    """Try exact match in dataframe."""
    query_df = df[(df["SoluteName"] == solute) & (df["Solvent"] == solvent)]
    if charge is not None:
        query_df = query_df[query_df["Charge"] == charge]
    if len(query_df) > 0:
        return query_df.iloc[0]["Predicted_DeltaGsolv"], solute, solvent
    return None

def similarity_search_fallback(query: str, k: int = 3) -> list:
    """Fallback to vector or BM25 search."""
    # Hybrid: Use vectorstore for semantic, BM25 for keyword
    semantic_docs = vectorstore.similarity_search(query, k=k)
    bm25_docs = bm25_retriever.get_relevant_documents(query)
    
    # Combine and dedup
    all_docs = semantic_docs + [doc for doc in bm25_docs if doc not in semantic_docs]
    return all_docs[:k]

# ==========================================================
# GROQ CLIENT (Refined with more prompt templates)
# ==========================================================
client = Groq(api_key=st.secrets["GROQ_API_KEY"])

PROMPT_TEMPLATES = {
    "explanation": """Provide exactly 5 concise scientific lines explaining the thermodynamic meaning and molecular interactions for the solvation free energy (ΔGsolv).
    Focus on: entropy/enthalpy contributions, solute-solvent interactions (e.g., H-bonding, dispersion), and implications for solubility.
    Solute: {solute}, Solvent: {solvent}, ΔGsolv: {deltag} kcal/mol.""",
    
    "concept": """Explain the concept of solvation free energy (ΔGsolv) in 4-6 bullet points. Cover: definition, calculation methods, factors influencing it (e.g., polarity, size), and applications in chemistry/pharma.""",
    
    "comparison": """Compare the solvation of {solute} in {solvent1} vs {solvent2}. Highlight differences in ΔGsolv values, molecular reasons, and practical implications. Keep to 5 lines.""",
    
    "help": """You are a helpful chemistry assistant. Provide usage examples for querying ΔGsolv, like "What is ΔG for Na+ in water?" or "Predict solvation of benzene in ethanol." End with a call to action.""",
    
    "error": """Gently guide the user to provide clear solute, solvent, and optional charge. Suggest examples and explain why specificity helps accuracy."""
}

def generate_response(prompt_template: str, **kwargs) -> str:
    """Generate response using Groq with templated prompts."""
    full_prompt = prompt_template.format(**kwargs)
    completion = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {"role": "system", "content": "You are a precise, scientific chemistry assistant. Respond concisely and accurately."},
            {"role": "user", "content": full_prompt}
        ],
        temperature=0.3,  # Lower for consistency
        max_tokens=300
    )
    return completion.choices[0].message.content.strip()

# ==========================================================
# CHAT MEMORY
# ==========================================================
if "messages" not in st.session_state:
    st.session_state.messages = []

if len(st.session_state.messages) == 0:
    st.markdown("""
    <div style="
        display:flex;
        justify-content:center;
        align-items:center;
        height:60vh;
        text-align:center;
        flex-direction:column;
    ">
        <h2>🧪 Welcome to MnSol ΔG Assistant</h2>
        <p>Ask about solvation free energy (ΔGsolv)</p>
        <p>Example: "What's the ΔG for Na+ in water?"</p>
    </div>
    """, unsafe_allow_html=True)

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ==========================================================
# CHAT INPUT (Refined with more intents and fallback)
# ==========================================================
prompt = st.chat_input("Ask about ΔGsolv... (e.g., 'Na+ in water')")

if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # -------- Intent Handling --------
    intent = detect_intent(prompt)
    
    if intent == "greeting":
        response = "👋 Hello! I am your MnSol ΔG Chemistry Assistant. Ready to dive into solvation free energies?"
    elif intent == "small_talk":
        response = "😊 I'm doing great—processing thermodynamic wonders! How can I assist with ΔGsolv today?"
    elif intent == "identity":
        response = generate_response(PROMPT_TEMPLATES["help"])  # Reuse help for intro
        response = "I am the **MnSol ΔG Chemistry Assistant** 🧪.\n\n" + response
    elif intent == "help":
        response = generate_response(PROMPT_TEMPLATES["help"])
    elif intent == "thanks":
        response = "You're welcome! 😊 Got more solvation queries?"
    elif intent == "goodbye":
        response = "Goodbye! 👋 Return anytime for ΔG insights."
    elif intent == "irrelevant":
        response = generate_response(PROMPT_TEMPLATES["error"])
    else:  # Query intent
        solute, solvent, charge = extract_entities(prompt)
        if solute and solvent:
            exact_result = try_exact_match(solute, solvent, charge)
            if exact_result:
                deltag, matched_solute, matched_solvent = exact_result
                explanation = generate_response(PROMPT_TEMPLATES["explanation"], solute=matched_solute, solvent=matched_solvent, deltag=deltag)
                response = f"""
🧪 **Predicted ΔGsolv**: {deltag} kcal/mol for **{matched_solute}** (charge: {charge if charge else 'neutral'}) in **{matched_solvent}**.

{chr(10).join(['• ' + line for line in explanation.split(chr(10)) if line.strip()])}
                """
            else:
                # Fallback to similarity
                fallback_docs = similarity_search_fallback(prompt, k=2)
                if fallback_docs:
                    response = "No exact match found. Here are similar predictions:\n\n"
                    for i, doc in enumerate(fallback_docs, 1):
                        meta = doc.metadata
                        response += f"{i}. **{meta['solute']}** (charge: {meta['charge']}) in **{meta['solvent']}**: {meta['deltag']} kcal/mol\n"
                    response += "\nRefine your query for exact results!"
                else:
                    response = generate_response(PROMPT_TEMPLATES["error"])
        else:
            response = generate_response(PROMPT_TEMPLATES["error"])
    
    with st.chat_message("assistant"):
        st.markdown(response)
    st.session_state.messages.append({"role": "assistant", "content": response})
