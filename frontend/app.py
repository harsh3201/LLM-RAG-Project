import streamlit as st
import requests
import json
import os

BACKEND_URL = "http://localhost:8000/api/v1"

st.set_page_config(
    page_title="RAG AI",
    page_icon="✨",
    layout="wide",
    initial_sidebar_state="expanded"
)

def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

if os.path.exists("style.css"):
    local_css("style.css")

with st.sidebar:
    st.markdown("## 🔮 Knowledge Hub")
    st.markdown("Upload documents to fuse them with the AI neural network.")
    
    uploaded_file = st.file_uploader("Drop Knowledge (PDF/TXT)", type=["pdf", "txt"])
    
    if uploaded_file is not None:
        if st.button("🚀 Initialize Ingestion"):
            with st.spinner("✨ Transmuting data into vectors..."):
                try:
                    files = {"file": (uploaded_file.name, uploaded_file, uploaded_file.type)}
                    response = requests.post(f"{BACKEND_URL}/upload", files=files)
                    
                    if response.status_code == 200:
                        data = response.json()
                        st.success(f"⚡ Neural Link Established: {data['chunks']} segments active.")
                    else:
                        st.error(f"❌ Ingestion Failed: {response.text}")
                except Exception as e:
                    st.error(f"🔌 Neural Network Disconnected: {e}")

    st.markdown("---")
    st.markdown("### 🧠 Artificial Intelligence")
    model_choice = st.radio(
        "Select Core Model:",
        options=["gemini", "openai"],
        format_func=lambda x: "✨ Google Gemini (Free)" if x == "gemini" else "🧠 OpenAI GPT-3.5 (Paid)",
        index=0
    )
    st.markdown("---")
    st.markdown("### ⚡ System Status")
    st.markdown(f"Engine: **{model_choice.upper()}**")
    st.markdown("Latency: **Ultra-Low**")

st.markdown("# ✨ Intelligent Neural Interface")
st.markdown("Query your data universe with context-aware generative intelligence.")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "sources" in message:
            with st.expander("🔍 Neural Traces (Sources)"):
                for src in message["sources"]:
                    st.markdown(f"**📄 Document**: {src['metadata'].get('source', 'Unknown')}")
                    st.markdown(f">_{src['page_content'][:300]}..._")

if prompt := st.chat_input("Input query parameter..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        
        with st.spinner("🧠 Synthesizing response..."):
            try:
                payload = {
                    "query": prompt, 
                    "top_k": 3,
                    "model_provider": model_choice
                }
                response = requests.post(f"{BACKEND_URL}/query", json=payload)
                
                if response.status_code == 200:
                    data = response.json()
                    answer = data.get("answer", "No intelligence retrieved.")
                    sources = data.get("sources", [])
                    
                    content = answer
                    if isinstance(content, list):
                        content = "".join([part if isinstance(part, str) else str(part) for part in content])
                    
                    full_response = content
                    message_placeholder.markdown(full_response)
                    
                    st.session_state.messages.append({
                        "role": "assistant", 
                        "content": full_response,
                        "sources": sources
                    })
                    
                    if sources:
                        with st.expander("🔍 Neural Traces"):
                            for src in sources:
                                st.markdown(f"**📄 Document**: {src['metadata'].get('source', 'Unknown')}")
                                st.markdown(f">_{src['page_content'][:300]}..._")
                
                else:
                    try:
                        err_json = response.json()
                        detail = err_json.get("detail", "")
                        if response.status_code == 401:
                            error_msg = "🚨 **Access Denied**: Invalid Neural Key. Check .env configuration."
                        elif response.status_code == 429 or "insufficient_quota" in str(detail) or "RESOURCE_EXHAUSTED" in str(detail):
                             error_msg = "💳 **Energy Depleted**: Quota exceeded. Recharge or wait for cool-down."
                        else:
                            if isinstance(detail, dict) and 'message' in detail:
                                error_msg = f"❌ Error: {detail['message']}"
                            else:
                                error_msg = f"❌ Error: {detail if detail else response.text}"
                    except:
                        error_msg = f"❌ Critical Error: {response.text}"
                    
                    message_placeholder.error(error_msg)
                    st.session_state.messages.append({"role": "assistant", "content": error_msg})

            except Exception as e:
                error_msg = f"🔌 Connection Terminated: {e}"
                message_placeholder.error(error_msg)
                st.session_state.messages.append({"role": "assistant", "content": error_msg})
