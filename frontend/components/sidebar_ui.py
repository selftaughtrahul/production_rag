import streamlit as st

def render_sidebar(api_client):
    with st.sidebar:
        st.header("Document Management")
        uploaded_file = st.file_uploader("Upload a document", type=["txt", "pdf", "md"])
        if uploaded_file is not None:
            if st.button("Upload"):
                with st.spinner("Uploading..."):
                    success, response = api_client.upload_document(uploaded_file.getvalue(), uploaded_file.name)
                    if success:
                        st.success(f"Document {uploaded_file.name} uploaded successfully!")
                    else:
                        st.error(f"Failed to upload: {response}")

        st.divider()

        st.header("Conversations")
        sessions = api_client.list_conversations()
        
        if st.button("New Chat"):
            st.session_state.current_session_id = None
            st.session_state.messages = []
            st.rerun()

        for session_id in sessions:
            if st.button(session_id[:8] + "...", key=f"session_{session_id}"):
                st.session_state.current_session_id = session_id
                messages = api_client.get_conversation(session_id)
                st.session_state.messages = messages
                st.rerun()

        st.divider()
        if st.button("Logout"):
            st.session_state.is_authenticated = False
            st.session_state.api_client.set_token(None)
            st.session_state.current_session_id = None
            st.session_state.messages = []
            st.rerun()
