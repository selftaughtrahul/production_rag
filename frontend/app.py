import streamlit as st
from api_client import APIClient
from components.auth_ui import render_auth_ui
from components.chat_ui import render_chat_ui
from components.sidebar_ui import render_sidebar

st.set_page_config(page_title="RAG Application", page_icon="🤖", layout="wide")

def init_session_state():
    if "api_client" not in st.session_state:
        st.session_state.api_client = APIClient()
    if "is_authenticated" not in st.session_state:
        st.session_state.is_authenticated = False
    if "current_session_id" not in st.session_state:
        st.session_state.current_session_id = None
    if "messages" not in st.session_state:
        st.session_state.messages = []

def main():
    init_session_state()
    
    if not st.session_state.is_authenticated:
        render_auth_ui(st.session_state.api_client)
    else:
        render_sidebar(st.session_state.api_client)
        render_chat_ui(st.session_state.api_client)

if __name__ == "__main__":
    main()
