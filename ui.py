import streamlit as st
from bot import answer_question

st.set_page_config(page_title="HR Assistant Bot",page_icon="🧑‍💼")
st.title("🧑‍💼 HR Assistant Chatbot")
st.markdown("Ask about company policies, benefits, or anything else related to HR.")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

user_input=st.chat_input("Ask a question abt HR..")

if user_input:
    st.session_state.chat_history.append({"role":"user","content":user_input})
    answer=answer_question(user_input)
    st.session_state.chat_history.append({"role":"assistant","content":answer})

for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])