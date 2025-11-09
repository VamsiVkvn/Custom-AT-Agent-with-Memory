import os
import uuid
import streamlit as st
from dotenv import load_dotenv
from assistant import (
    generate_assistant_response, 
    update_long_term_memory, 
    transcribe_audio, 
    clear_memory
)

load_dotenv()

st.set_page_config(page_title="Custom AI Assistant with Memory")
st.title("🗣️ AI Voice/Text Assistant with Long-Term Memory")

if "messages" not in st.session_state:
    st.session_state.messages = []
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

with st.sidebar:
    st.header("Controls")
    if st.button("Clear Conversation & Memory"):
        clear_memory()
        st.session_state.messages = []
        st.session_state.session_id = str(uuid.uuid4())
        st.experimental_rerun()
    st.caption(f"Current Session ID: {st.session_state.session_id[:8]}...")
    st.caption("Long-term memory is stored in ChromaDB.")

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])



if prompt := st.chat_input("Type your message here..."):
    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.spinner("Thinking..."):
        full_history = [
            {"role": m["role"], "content": m["content"]}
            for m in st.session_state.messages if m["role"] in ["user", "assistant"]
        ]
        assistant_response = generate_assistant_response(prompt, full_history)

    with st.chat_message("assistant"):
        st.markdown(assistant_response)

    st.session_state.messages.append({"role": "assistant", "content": assistant_response})

    update_long_term_memory(
        user_input=prompt,
        assistant_response=assistant_response,
        conversation_id=st.session_state.session_id
    )
