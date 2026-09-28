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

        for conversation in sessions:
            session_id = conversation["session_id"]
            raw_message = (conversation.get("first_message") or "").strip()
            label = raw_message or "New conversation"
            if len(label) > 40:
                label = label[:40] + "..."

            title_col, delete_col = st.columns([5, 1], vertical_alignment="center")
            with title_col:
                if st.button(
                    label,
                    key=f"session_{session_id}",
                    use_container_width=True,
                ):
                    st.session_state.current_session_id = session_id
                    st.session_state.messages = api_client.get_conversation(session_id)
                    st.rerun()
            if raw_message:
                with delete_col:
                    if st.button(
                        "✕",
                        key=f"delete_{session_id}",
                        help="Delete this conversation",
                    ):
                        deleted, error = api_client.delete_conversation(session_id)
                        if deleted:
                            if st.session_state.current_session_id == session_id:
                                st.session_state.current_session_id = None
                                st.session_state.messages = []
                            st.rerun()
                        st.error(error or "Could not delete this conversation")

        st.divider()
        if st.button("Logout"):
            st.session_state.is_authenticated = False
            st.session_state.access_token = None
            st.session_state.api_client.set_token(None)
            st.session_state.current_session_id = None
            st.session_state.messages = []
            st.rerun()
