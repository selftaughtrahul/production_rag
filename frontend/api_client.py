import requests
import json
import streamlit as st

BASE_URL = "http://localhost:8000"

_CHAT_ROLES = {"human": "user", "ai": "assistant", "user": "user", "assistant": "assistant"}


class APIClient:
    def __init__(self):
        self.token = None
        self.last_session_id = None

    def set_token(self, token: str | None):
        self.token = token
        st.session_state.access_token = token

    def get_headers(self):
        token = self.token or st.session_state.get("access_token")
        if token and not self.token:
            self.token = token
        headers = {}
        if token:
            headers["Authorization"] = f"Bearer {token}"
        return headers

    def _check_auth(self, response):
        if response.status_code == 401:
            st.session_state.is_authenticated = False
            st.session_state.access_token = None
            self.token = None

    def login(self, username, password):
        response = requests.post(
            f"{BASE_URL}/auth/login",
            data={"username": username, "password": password}
        )
        if response.status_code == 200:
            data = response.json()
            self.set_token(data.get("access_token"))
            return True, "Login successful"
        return False, response.text

    def register(self, username, email, password):
        response = requests.post(
            f"{BASE_URL}/auth/register",
            json={"username": username, "email": email, "password": password}
        )
        if response.status_code == 200 or response.status_code == 201:
            return True, "Registration successful"
        return False, response.text

    def get_me(self):
        response = requests.get(f"{BASE_URL}/auth/me", headers=self.get_headers())
        self._check_auth(response)
        if response.status_code == 200:
            return response.json()
        return None

    def upload_document(self, file_bytes, filename):
        files = {"file": (filename, file_bytes)}
        response = requests.post(
            f"{BASE_URL}/ingest",
            headers=self.get_headers(),
            files=files
        )
        self._check_auth(response)
        return response.status_code == 200, response.json() if response.status_code == 200 else response.text

    def list_documents(self):
        response = requests.get(f"{BASE_URL}/documents", headers=self.get_headers())
        self._check_auth(response)
        if response.status_code == 200:
            return response.json()
        return []

    def list_conversations(self):
        response = requests.get(f"{BASE_URL}/chat/conversations", headers=self.get_headers())
        self._check_auth(response)
        if response.status_code == 200:
            return response.json().get("sessions", [])
        return []

    def get_conversation(self, session_id):
        response = requests.get(f"{BASE_URL}/chat/conversations/{session_id}", headers=self.get_headers())
        self._check_auth(response)
        if response.status_code == 200:
            return [
                {
                    "role": _CHAT_ROLES.get(m.get("role", ""), m.get("role", "assistant")),
                    "content": m.get("content", ""),
                }
                for m in response.json().get("messages", [])
            ]
        return []

    def query_stream(
        self,
        question: str,
        session_id: str = None,
        *,
        access_token: str | None = None,
    ):
        """Yields text chunks as they arrive from the single chat endpoint."""
        payload = {"question": question}
        if session_id:
            payload["session_id"] = session_id

        # Capture the JWT on the script thread. st.write_stream may iterate this
        # generator later, when session_state is empty and self.token is stale.
        token = access_token or self.token or st.session_state.get("access_token")
        if not token:
            yield "You are not logged in. Please log in again."
            return

        headers = {
            "Authorization": f"Bearer {token}",
            "Accept": "text/event-stream",
        }
        http = requests.Session()
        http.headers.update({"Authorization": f"Bearer {token}"})
        try:
            response = http.post(
                f"{BASE_URL}/chat/",
                headers=headers,
                json=payload,
                stream=True,
                timeout=(15, 600),
            )
            if response.status_code in {301, 302, 307, 308}:
                location = response.headers.get("Location")
                if location:
                    response = http.post(
                        location,
                        headers=headers,
                        json=payload,
                        stream=True,
                        timeout=(15, 600),
                    )
        finally:
            http.close()

        if response.status_code == 401:
            yield "Your session expired. Please log in again."
            return
        if response.status_code != 200:
            yield f"Error: {response.text}"
            return
            
        for line in response.iter_lines():
            if line:
                line_str = line.decode("utf-8")
                if line_str.startswith("data: "):
                    data_str = line_str[6:]
                    try:
                        event = json.loads(data_str)
                        if event.get("type") == "token":
                            yield event.get("content", "")
                        elif event.get("type") == "done":
                            self.last_session_id = event.get("session_id")
                        elif event.get("type") == "error":
                            yield f"\n\nError: {event.get('error')}"
                    except json.JSONDecodeError:
                        pass
