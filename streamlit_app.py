from __future__ import annotations

import httpx
import streamlit as st


st.set_page_config(page_title="RAG Documentation Assistant", layout="wide")

DEFAULT_API_URL = "http://127.0.0.1:8000"


def create_session(api_base_url: str) -> str:
    response = httpx.post(f"{api_base_url}/session/create", timeout=30.0)
    response.raise_for_status()
    payload = response.json()
    return payload["session_id"]


def fetch_session(api_base_url: str, session_id: str) -> dict:
    response = httpx.get(f"{api_base_url}/session/{session_id}", timeout=30.0)
    response.raise_for_status()
    return response.json()


def run_query(api_base_url: str, question: str, session_id: str | None) -> dict:
    response = httpx.post(
        f"{api_base_url}/query",
        json={"question": question, "session_id": session_id},
        timeout=120.0,
    )
    response.raise_for_status()
    return response.json()


def clear_session(api_base_url: str, session_id: str) -> None:
    response = httpx.delete(f"{api_base_url}/session/{session_id}", timeout=30.0)
    response.raise_for_status()


if "api_base_url" not in st.session_state:
    st.session_state.api_base_url = DEFAULT_API_URL
if "session_id" not in st.session_state:
    st.session_state.session_id = None
if "messages" not in st.session_state:
    st.session_state.messages = []
if "last_response" not in st.session_state:
    st.session_state.last_response = None


st.title("RAG-Based Technical Documentation Assistant")

left_col, right_col = st.columns([3, 2], gap="large")

with right_col:
    st.subheader("Session Information")
    api_base_url = st.text_input("FastAPI Base URL", value=st.session_state.api_base_url)
    st.session_state.api_base_url = api_base_url.rstrip("/")

    action_col_1, action_col_2 = st.columns(2)
    if action_col_1.button("New Session", use_container_width=True):
        try:
            st.session_state.session_id = create_session(st.session_state.api_base_url)
            st.session_state.messages = []
            st.session_state.last_response = None
            st.success("Created a new session.")
        except Exception as exc:
            st.error(f"Could not create session: {exc}")

    if action_col_2.button("Clear Session", use_container_width=True):
        if st.session_state.session_id:
            try:
                clear_session(st.session_state.api_base_url, st.session_state.session_id)
                st.session_state.session_id = None
                st.session_state.messages = []
                st.session_state.last_response = None
                st.success("Session cleared.")
            except Exception as exc:
                st.error(f"Could not clear session: {exc}")

    st.caption(f"Session ID: {st.session_state.session_id or 'Not created yet'}")
    if st.session_state.session_id:
        try:
            session = fetch_session(st.session_state.api_base_url, st.session_state.session_id)
            st.write(f"Created: {session['created_at']}")
            st.write(f"Updated: {session['updated_at']}")
            st.write(f"Messages Stored: {len(session['chat_history'])}")
        except Exception as exc:
            st.warning(f"Could not load session metadata: {exc}")

    st.divider()
    st.subheader("Source Citations Panel")
    response = st.session_state.last_response
    if response:
        hallucination = response["hallucination_check"]
        grounded_label = "Grounded" if hallucination["grounded"] else "Not Grounded"
        st.write(f"Hallucination Check Status: {grounded_label}")
        st.write(f"Confidence Score: {hallucination['confidence_score']:.2f}")
        st.write(f"Explanation: {hallucination['explanation']}")
        if response.get("warning"):
            st.warning(response["warning"])
        st.write(f"Used Web Search: {'Yes' if response.get('used_web_search') else 'No'}")
        for source in response.get("sources", []):
            badge = "Documentation" if source["source_kind"] == "local_document" else "Web Search"
            st.markdown(f"**{source['document_name']}**")
            st.caption(f"{badge} | {source['location']}")
            if source.get("snippet"):
                st.write(source["snippet"])
            st.divider()
    else:
        st.info("Ask a question to see citations and grounding status.")

with left_col:
    st.subheader("Chat Interface")
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    prompt = st.chat_input("Ask about the indexed docs or use a follow-up question")
    if prompt:
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        try:
            result = run_query(
                st.session_state.api_base_url,
                prompt,
                st.session_state.session_id,
            )
            st.session_state.session_id = result["session_id"]
            st.session_state.last_response = result
            st.session_state.messages.append({"role": "assistant", "content": result["answer"]})
            with st.chat_message("assistant"):
                st.markdown(result["answer"])
        except Exception as exc:
            error_message = f"Query failed: {exc}"
            st.session_state.messages.append({"role": "assistant", "content": error_message})
            with st.chat_message("assistant"):
                st.error(error_message)
