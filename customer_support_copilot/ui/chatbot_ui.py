import streamlit as st
import requests
import time

st.set_page_config(page_title="Support Copilot", page_icon="🎧")
st.title("🎧 Enterprise AI Support Copilot")
st.caption("Fine-Tuned Llama-3 (8B) + RAG Retrieval Pipeline")

with st.sidebar:
    st.header(" Architecture")
    st.markdown("- **Model:** Llama-3 (8B QLoRA)")
    st.markdown("- **Data:** Bitext Customer Support")
    st.success(" Backend Connected (FastAPI)")

if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Hello! I am your AI Support Copilot. How can I help?"}]

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("E.g., Where is my refund?!"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Retrieving Knowledge Base..."):
            context = "KB_ARTICLE_402: Refunds take 3-5 business days." if "refund" in prompt.lower() else "KB_ARTICLE_101: General greeting."
            
            try:
                res = requests.post("http://localhost:8000/chat", json={"query": prompt, "context": context}, timeout=5)
                res.raise_for_status()
                ai_response = res.json()["response"]
            except requests.exceptions.RequestException:
                ai_response = "⚠️ System Error: Ensure FastAPI (src/app.py) is running on port 8000."

            with st.expander("🔍 View Retrieved Knowledge Base Context"):
                st.info(context)
            
            placeholder = st.empty()
            full_response = ""
            for chunk in ai_response.split():
                full_response += chunk + " "
                time.sleep(0.05)
                placeholder.markdown(full_response + "▌")
            placeholder.markdown(full_response)
            
    st.session_state.messages.append({"role": "assistant", "content": ai_response})