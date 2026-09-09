import streamlit as st

def render_auth_ui(api_client):
    st.title("Login to RAG Application")
    
    tab1, tab2 = st.tabs(["Login", "Register"])
    
    with tab1:
        st.header("Login")
        username = st.text_input("Email", key="login_username")
        password = st.text_input("Password", type="password", key="login_password")
        if st.button("Login"):
            success, msg = api_client.login(username, password)
            if success:
                st.session_state.is_authenticated = True
                st.success("Logged in successfully!")
                st.rerun()
            else:
                st.error(f"Login failed: {msg}")

    with tab2:
        st.header("Register")
        reg_username = st.text_input("Username", key="reg_username")
        reg_email = st.text_input("Email", key="reg_email")
        reg_password = st.text_input("Password", type="password", key="reg_password")
        if st.button("Register"):
            success, msg = api_client.register(reg_username, reg_email, reg_password)
            if success:
                st.success("Registered successfully! You can now login.")
            else:
                st.error(f"Registration failed: {msg}")
