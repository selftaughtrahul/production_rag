import streamlit as st

def render_chat_ui(api_client):
    st.title("Chat with your Documents")
    
    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # React to user input
    if prompt := st.chat_input(
        "Ask a question about your documents...",
        submit_mode="disable",
    ):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                full_response = st.write_stream(
                    api_client.query_stream(
                        prompt,
                        st.session_state.current_session_id,
                        access_token=st.session_state.get("access_token")
                        or api_client.token,
                    )
                )

        if api_client.last_session_id:
            st.session_state.current_session_id = api_client.last_session_id

        st.session_state.messages.append(
            {"role": "assistant", "content": full_response or ""}
        )
