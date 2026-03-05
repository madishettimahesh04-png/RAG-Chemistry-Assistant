# ==========================================================
# 🤖 MnSol Conversational ΔG Assistant (Production Version)
# ==========================================================

import os
import re
import gdown
import zipfile
import streamlit as st
import pandas as pd

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from groq import Groq
from rapidfuzz import process, fuzz


# ------------------------------------------------
# UI CONFIG
# ------------------------------------------------

st.set_page_config(page_title="MnSol AI Assistant", layout="centered")

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
# INTENT DETECTION
# ==========================================================

def is_greeting(text):
    greetings = ["hi","hello","hey","hi there","good morning","good afternoon","good evening"]
    text = text.lower()
    return any(word in text for word in greetings)


def is_small_talk(text):
    small_talk = ["how are you","how are you doing","how's it going","what's up"]
    text = text.lower()
    return any(word in text for word in small_talk)


def is_capability_question(text):
    capability_words = [
        "what can you do",
        "your capabilities",
        "features",
        "about this chatbot",
        "about you",
        "what do you do",
        "how can you help",
        "mnsol ai",
        "about mnsol assistant"
    ]
    text = text.lower()
    return any(word in text for word in capability_words)


def is_identity_question(text):
    identity_phrases = ["your name","who are you","what is your name","tell me about yourself"]
    text = text.lower()
    return any(word in text for word in identity_phrases)


def is_thanks(text):
    thanks_words = ["thank you","thanks","thx"]
    text = text.lower()
    return any(word in text for word in thanks_words)


def is_goodbye(text):
    goodbye_words = ["bye","goodbye","see you"]
    text = text.lower()
    return any(word in text for word in goodbye_words)


def is_irrelevant(text):
    chemistry_keywords = [
        "solvent","solute","delta","deltag",
        "water","methanol","ethanol","ion","molecule","benzene"
    ]
    text = text.lower()
    return not any(word in text for word in chemistry_keywords)


# ==========================================================
# DATA LOADING
# ==========================================================

FILE_ID = "1gUKTTKNjOqI2jP3I6bGVSmFtb3JD7jn2"
ZIP_FILE = "mnsol_faiss_index.zip"
INDEX_FOLDER = "mnsol_faiss_index"

if not os.path.exists(INDEX_FOLDER):

    with st.spinner("Downloading vector database..."):

        url = f"https://drive.google.com/uc?id={FILE_ID}"

        gdown.download(url, ZIP_FILE, quiet=False)

        with zipfile.ZipFile(ZIP_FILE, "r") as zip_ref:
            zip_ref.extractall(".")


df = pd.read_csv("five_columns_dataset_with_predictions_3.csv")

solute_list = df["SoluteName"].astype(str).unique().tolist()
solvent_list = df["Solvent"].astype(str).unique().tolist()


# ==========================================================
# FUZZY MATCHING
# ==========================================================

def resolve_solute(user_input):

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


# ==========================================================
# QUERY PARSER
# ==========================================================

def try_exact_from_query(text):

    found_solute = None
    found_solvent = None
    found_charge = None

    charge_match = re.search(r"([+-]?\d+)", text)

    if charge_match:
        try:
            found_charge = int(charge_match.group())
        except:
            pass

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


# ==========================================================
# GROQ LLM
# ==========================================================

client = Groq(api_key=st.secrets["GROQ_API_KEY"])


# ==========================================================
# CHAT MEMORY
# ==========================================================

if "messages" not in st.session_state:
    st.session_state.messages = []

MAX_MEMORY = 30

def trim_memory():

    if len(st.session_state.messages) > MAX_MEMORY:
        st.session_state.messages = st.session_state.messages[-MAX_MEMORY:]


# ==========================================================
# WELCOME SCREEN
# ==========================================================

if len(st.session_state.messages) == 0:

    st.markdown("""
    <div style="display:flex;justify-content:center;align-items:center;height:60vh;text-align:center;flex-direction:column;">
        <h2>🧪 Welcome to MnSol ΔG Assistant</h2>
        <p>Ask about solvation free energy (ΔGsolv)</p>
    </div>
    """, unsafe_allow_html=True)


for msg in st.session_state.messages:

    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])


# ==========================================================
# CHAT INPUT
# ==========================================================

prompt = st.chat_input("Ask about ΔGsolv...")

if prompt:

    st.session_state.messages.append({"role":"user","content":prompt})
    trim_memory()

    with st.chat_message("user"):
        st.markdown(prompt)

    # ------------------------------------------------
    # INTENT ROUTING
    # ------------------------------------------------

    if is_greeting(prompt):

        response = "👋 Hello! I am your MnSol ΔG Chemistry Assistant."

    elif is_small_talk(prompt):

        response = "😊 I'm doing great! How can I help with solvation free energy?"

    elif is_identity_question(prompt):

        response = "I am an AI assistant specialized in predicting and explaining solvation free energy (ΔGsolv)."

    elif is_thanks(prompt):

        response = "You're welcome! 😊 Let me know the solute, solvent and charge."

    elif is_goodbye(prompt):

        response = "Goodbye! 👋 Feel free to return for ΔG analysis anytime."

    elif is_capability_question(prompt):

        response = """
### 🧪 MnSol AI Solvation Assistant

I help researchers analyze **solvation free energy (ΔGsolv)** using the **MnSol dataset and AI models**.

### 🔬 Capabilities

• Predict ΔGsolv for solute–solvent systems  
• Explain thermodynamic meaning of ΔG values  
• Analyze molecular interactions in solution  
• Identify molecules using fuzzy matching  
• Provide scientific explanations using AI  

### Example Questions

• ΔG of benzene in water  
• Solvation energy of ethanol in methanol  
• Explain ΔG for acetone in water
"""

    # ------------------------------------------------
    # DATASET SEARCH
    # ------------------------------------------------

    else:

        deltag, solute, solvent = try_exact_from_query(prompt)

        if deltag is not None:

            base = f"""
The predicted solvation free energy (ΔGsolv) of **{solute}**
in **{solvent}** is **{deltag} kcal/mol**.
"""

            explanation = client.chat.completions.create(

                model="llama-3.1-8b-instant",

                messages=[
                    {
                        "role":"system",
                        "content":"Provide exactly 5 concise scientific lines explaining the thermodynamic meaning and molecular interactions."
                    },
                    {
                        "role":"user",
                        "content":f"Solute: {solute}, Solvent: {solvent}, DeltaG: {deltag}"
                    }
                ]
            )

            response = base + "\n\n" + explanation.choices[0].message.content

        else:

            llm_response = client.chat.completions.create(

                model="llama-3.1-8b-instant",

                messages=[
                    {
                        "role":"system",
                        "content":"You are a computational chemistry assistant specializing in solvation thermodynamics."
                    },
                    {
                        "role":"user",
                        "content":prompt
                    }
                ]
            )

            response = llm_response.choices[0].message.content


    # ------------------------------------------------
    # DISPLAY RESPONSE
    # ------------------------------------------------

    with st.chat_message("assistant"):
        st.markdown(response)

    st.session_state.messages.append({"role":"assistant","content":response})
    trim_memory()
