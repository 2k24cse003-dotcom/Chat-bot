"""
Streamlit app that uses IBM watsonx.ai (e.g. granite-3-2b-instruct) to build a Personal Finance Chatbot.

Usage:
  1. pip install -r requirements.txt
     (requirements: streamlit, requests)
  2. Set environment variables:
       WATSONX_API_KEY  - your IBM Cloud API key (IAM apikey)
       WATSONX_URL      - your watsonx.ai endpoint, e.g. https://api.us-south.watsonx.ai
       WATSONX_MODEL    - optional, default 'granite-3-2b-instruct'
  3. Run: streamlit run streamlit_ibm_personal_finance_app.py

Notes:
  - This example uses the watsonx.ai REST generation endpoint (obtain IAM token then call generation).
  - For production use: add robust error handling, rate-limit/backoff, input validation and cost controls.
"""

import os
import time
import json
import requests
import streamlit as st
from typing import Optional

# ---------- Configuration ----------
API_KEY = os.environ.get("WATSONX_API_KEY")
WATSONX_URL = os.environ.get("WATSONX_URL")
MODEL_ID = os.environ.get("WATSONX_MODEL", "granite-3-2b-instruct")
IAM_TOKEN_URL = "https://iam.cloud.ibm.com/identity/token"
API_VERSION = "2024-05-31"  # version param for generation endpoint

# ---------- Helper functions ----------
@st.cache_resource
def get_iam_token(apikey: str) -> Optional[dict]:
    """Exchange an IBM Cloud API key for a bearer token. Returns dict with token and expiry.
    Token typically expires in 1 hour.
    """
    if not apikey:
        return None
    data = {
        "grant_type": "urn:ibm:params:oauth:grant-type:apikey",
        "apikey": apikey
    }
    headers = {"Content-Type": "application/x-www-form-urlencoded"}
    try:
        r = requests.post(IAM_TOKEN_URL, data=data, headers=headers, timeout=10)
        r.raise_for_status()
        resp = r.json()
        # resp contains access_token and expires_in (seconds)
        return {"access_token": resp.get("access_token"), "expires_in": resp.get("expires_in"), "fetched_at": time.time()}
    except Exception as e:
        st.error(f"Failed to obtain IAM token: {e}")
        return None


def is_token_valid(token_info: dict) -> bool:
    if not token_info: return False
    fetched = token_info.get("fetched_at", 0)
    expires = token_info.get("expires_in", 0)
    # add small buffer (30s)
    return (time.time() - fetched) < (expires - 30)


def ensure_token() -> Optional[str]:
    """Get or refresh IAM token and store it in session state."""
    if "watsonx_token" not in st.session_state or not is_token_valid(st.session_state["watsonx_token"]):
        token_info = get_iam_token(API_KEY)
        if not token_info:
            return None
        st.session_state["watsonx_token"] = token_info
    return st.session_state["watsonx_token"]["access_token"]


def call_watsonx_generate(prompt: str, max_tokens: int = 300, temperature: float = 0.2) -> str:
    """Call the watsonx generation endpoint for MODEL_ID with the provided prompt.
    This uses the generic /v1/generation/{model}/actions endpoint pattern.
    """
    token = ensure_token()
    if not token:
        return "Error: no IAM token available. Set WATSONX_API_KEY environment variable."
    if not WATSONX_URL:
        return "Error: WATSONX_URL environment variable not set."

    url = f"{WATSONX_URL}/v1/generation/{MODEL_ID}/actions?version={API_VERSION}"
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json"
    }
    payload = {
        "input": prompt,
        "parameters": {
            "max_new_tokens": max_tokens,
            "temperature": temperature,
            # you can add other model-specific parameters here
        }
    }
    try:
        resp = requests.post(url, headers=headers, json=payload, timeout=60)
        resp.raise_for_status()
        j = resp.json()
        # The response schema can vary depending on model/SDK; try to extract text generically
        # Common: j['output'][0]['content'][0]['text'] or j['generated_text'] or j['outputs']
        # We attempt a few known locations.
        if isinstance(j, dict):
            # attempt typical keys
            if "output" in j:
                # output is often a list of content blocks
                out = j["output"]
                # find the first text block
                if isinstance(out, list) and out:
                    for block in out:
                        if isinstance(block, dict) and "content" in block:
                            for c in block["content"]:
                                if isinstance(c, dict) and "text" in c:
                                    return c["text"]
            if "generated_text" in j:
                return j["generated_text"]
            if "outputs" in j and isinstance(j["outputs"], list) and j["outputs"]:
                first = j["outputs"][0]
                if isinstance(first, dict) and "text" in first:
                    return first["text"]
        # fallback: pretty print JSON
        return json.dumps(j, indent=2)
    except Exception as e:
        return f"Generation error: {e}"

# ---------- Streamlit UI ----------
st.set_page_config(page_title="Personal Finance Chatbot — IBM watsonx + Streamlit", layout="wide")
st.title("Personal Finance Chatbot")
st.markdown("Ask budgeting, savings, tax (general) and investment questions. Select persona to control tone and complexity.")

# Sidebar controls
with st.sidebar:
    st.header("Configuration")
    persona = st.selectbox("Persona / Tone", ["Student (simple, friendly)", "Professional (concise, technical)"])
    max_tokens = st.slider("Max tokens / response length", 100, 1200, 350, step=50)
    temp = st.slider("Temperature (creativity)", 0.0, 1.0, 0.2, step=0.05)
    st.divider()
    st.caption("Model: " + MODEL_ID)
    if st.button("Test connection to watsonx"):
        token = ensure_token()
        if token:
            st.success("IAM token acquired — generation calls should work (if WATSONX_URL is set).")
        else:
            st.error("Failed to get IAM token. Make sure WATSONX_API_KEY env var is set.")

# Main interaction
col1, col2 = st.columns([2,1])
with col1:
    user_prompt = st.text_area("Ask a question or paste your monthly expenses (or both):", height=200)
    if st.button("Get Advice"):
        if not user_prompt.strip():
            st.warning("Please type a question or paste expenses first.")
        else:
            # Compose system prompt with persona and few-shot instructs and user's input
            system_preamble = (
                "You are a helpful personal finance assistant. Provide clear, actionable advice. "
                "If the user provides a list of monthly expenses, produce a budget summary, suggest 3 ways to save, and an estimated savings goal."
            )
            if "Student" in persona:
                tone = "Use a friendly, simple tone suitable for a college student. Provide examples and keep explanations short."
            else:
                tone = "Use a concise, professional tone suitable for a working professional. Provide clear steps and numbers where possible."

            final_prompt = f"{system_preamble}\nTone: {tone}\nUser input:\n{user_prompt}\nRespond with bullet points where appropriate."
            with st.spinner("Contacting watsonx.ai..."):
                response_text = call_watsonx_generate(final_prompt, max_tokens=max_tokens, temperature=temp)
            st.subheader("Assistant response")
            st.markdown(response_text)

with col2:
    st.subheader("Quick helpers")
    if st.button("Sample: Build a monthly budget from salary"):
        example = (
            "I earn INR 40,000/month. Rent 12,000, food 6,000, transport 2,000, subscriptions 1,000, misc 3,000. I want to save for a laptop costing 50,000 in 9 months."
        )
        st.session_state['example_input'] = example
        st.experimental_rerun()
    if 'example_input' in st.session_state:
        st.info(st.session_state['example_input'])

    st.markdown("---")
    st.markdown("*Notes:*")
    st.markdown("- This app calls your IBM watsonx.ai instance — you will be billed according to IBM pricing for model inference. See your IBM account.")
    st.markdown("- This prototype is for educational/demo use. Do not rely on tax/legal advice from the model; verify with a professional.")

# Footer: show env var warnings
if not API_KEY:
    st.error("Missing WATSONX_API_KEY environment variable — put your IBM Cloud API key in WATSONX_API_KEY.")
if not WATSONX_URL:
    st.error("Missing WATSONX_URL environment variable — set to your watsonx.ai API base URL (e.g. https://api.region.watsonx.ai).")

# End of file
