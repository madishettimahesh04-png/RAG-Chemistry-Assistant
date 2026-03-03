# ==========================================================
# 🧪 MnSol Autonomous AI Agent (Real Architecture)
# ==========================================================

import os
import json
import re
from typing import Dict, Optional, List

import streamlit as st
import pandas as pd
from groq import Groq
from rapidfuzz import process, fuzz


# ==========================================================
# CONFIG
# ==========================================================

st.set_page_config(page_title="MnSol AI Agent", layout="wide")

st.title("🧪 MnSol Autonomous AI Agent")

client = Groq(api_key=st.secrets.get("GROQ_API_KEY"))

# ==========================================================
# LOAD DATA
# ==========================================================

@st.cache_resource
def load_data():
    df = pd.read_csv("five_columns_dataset_with_predictions_3.csv")
    return df

df = load_data()
solute_list = df["SoluteName"].astype(str).unique().tolist()
solvent_list = df["Solvent"].astype(str).unique().tolist()

# ==========================================================
# MEMORY
# ==========================================================

if "messages" not in st.session_state:
    st.session_state.messages = []

if "context" not in st.session_state:
    st.session_state.context = {}

# ==========================================================
# ===================== AI LAYERS ==========================
# ==========================================================

# ----------------------------------------------------------
# 1️⃣ INTENT CLASSIFIER (LLM)
# ----------------------------------------------------------

def classify_intent(query: str) -> str:

    prompt = f"""
Classify this query into one of:

- greeting
- identity
- concept
- comparison
- dataset_query
- irrelevant

Query: "{query}"

Return only the label.
"""

    completion = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {"role": "system", "content": "You are an intent classifier."},
            {"role": "user", "content": prompt}
        ],
        temperature=0
    )

    return completion.choices[0].message.content.strip().lower()


# ----------------------------------------------------------
# 2️⃣ ENTITY EXTRACTION (LLM STRUCTURED)
# ----------------------------------------------------------

def extract_entities(query: str) -> Dict:

    prompt = f"""
Extract chemistry entities from:

Query: "{query}"

Return strictly JSON:

{{
  "solute": "",
  "solvents": [],
  "charge": null
}}
"""

    completion = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {"role": "system", "content": "Extract structured chemistry data."},
            {"role": "user", "content": prompt}
        ],
        temperature=0
    )

    try:
        return json.loads(completion.choices[0].message.content)
    except:
        return {"solute": None, "solvents": [], "charge": None}


# ----------------------------------------------------------
# 3️⃣ TOOL LAYER
# ----------------------------------------------------------

class DatasetTool:

    def get_exact(self, solute: str, solvent: str, charge: Optional[int]):

        q = df[
            (df["SoluteName"] == solute) &
            (df["Solvent"] == solvent)
        ]

        if charge is not None:
            q = q[q["Charge"] == charge]

        if not q.empty:
            return float(q.iloc[0]["Predicted_DeltaGsolv"])

        return None


class ComparisonTool:

    def compare(self, solute: str, solvents: List[str], charge: Optional[int]):

        results = {}

        for solvent in solvents:
            q = df[
                (df["SoluteName"] == solute) &
                (df["Solvent"] == solvent)
            ]
            if charge is not None:
                q = q[q["Charge"] == charge]

            if not q.empty:
                results[solvent] = float(q.iloc[0]["Predicted_DeltaGsolv"])

        return results


dataset_tool = DatasetTool()
comparison_tool = ComparisonTool()

# ----------------------------------------------------------
# 4️⃣ REASONING ENGINE
# ----------------------------------------------------------

def reasoning_engine(data_context: str, user_query: str) -> str:

    prompt = f"""
You are an expert physical chemist.

Data:
{data_context}

User Question:
{user_query}

Provide step-by-step scientific reasoning and final answer.
"""

    completion = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {"role": "system", "content": "Scientific reasoning assistant."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.2
    )

    return completion.choices[0].message.content.strip()


# ----------------------------------------------------------
# 5️⃣ ORCHESTRATOR (AI BRAIN)
# ----------------------------------------------------------

def orchestrator(user_query: str) -> str:

    intent = classify_intent(user_query)
    entities = extract_entities(user_query)

    solute = entities.get("solute")
    solvents = entities.get("solvents", [])
    charge = entities.get("charge")

    # Store context
    if solute:
        st.session_state.context["last_solute"] = solute
    if solvents:
        st.session_state.context["last_solvents"] = solvents

    # ---------- ROUTING ----------

    if intent == "greeting":
        return "👋 Welcome to MnSol Autonomous AI Agent."

    if intent == "identity":
        return "🧪 I am an autonomous AI system for solvation thermodynamics."

    if intent == "concept":
        return reasoning_engine("Concept explanation requested.", user_query)

    if intent == "comparison" and solute and len(solvents) >= 2:

        results = comparison_tool.compare(solute, solvents, charge)

        if results:
            context_data = f"Comparison data: {results}"
            return reasoning_engine(context_data, user_query)

        return "⚠️ Could not find comparison data."

    if intent == "dataset_query" and solute and solvents:

        solvent = solvents[0]
        deltag = dataset_tool.get_exact(solute, solvent, charge)

        if deltag is not None:
            context_data = f"Solute: {solute}, Solvent: {solvent}, ΔG: {deltag}"
            return reasoning_engine(context_data, user_query)

        return "⚠️ No exact dataset match found."

    return "⚠️ Please provide valid solute and solvent information."


# ==========================================================
# CHAT UI
# ==========================================================

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("Ask about ΔGsolv..."):

    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("user"):
        st.markdown(prompt)

    response = orchestrator(prompt)

    with st.chat_message("assistant"):
        st.markdown(response)

    st.session_state.messages.append({"role": "assistant", "content": response})
