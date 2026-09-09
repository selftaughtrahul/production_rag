import streamlit as st

def render_chat_ui(api_client):
    st.title("Chat with your Documents")
    
    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # React to user input
    if prompt := st.chat_input("Ask a question about your documents..."):
        # Display user message in chat message container
        st.chat_message("user").markdown(prompt)
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})

        # Display assistant response in chat message container
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = ""
            
            # Stream the response from the API
            for chunk in api_client.query_stream(
                prompt,
                st.session_state.current_session_id,
                st.session_state.chat_mode,
            ):
                full_response += chunk
                message_placeholder.markdown(full_response + "▌")
                
            message_placeholder.markdown(full_response)

        if api_client.last_session_id:
            st.session_state.current_session_id = api_client.last_session_id

        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": full_response})
