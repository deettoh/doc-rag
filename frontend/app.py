"""DocRAG Streamlit Frontend - Main Application."""

import os
import time

import httpx
import streamlit as st

# Configuration
BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")
API_BASE = f"{BACKEND_URL}/api"
REQUEST_TIMEOUT = 60.0


st.set_page_config(
    page_title="DocRAG",
    layout="wide",
)


# region -------------------- Shared CSS --------------------
st.markdown(
    """
    <style>
    /* Main background & font */
    .block-container { padding-top: 2rem; }
    h1 { letter-spacing: -0.5px; }

    /* Hides the streamlit status icons in the top right */
    [data-testid="stStatusWidget"] { visibility: hidden; }

    /* Score badge colors */
    .score-high  { color: #22c55e; font-weight: 700; }
    .score-mid   { color: #eab308; font-weight: 700; }
    .score-low   { color: #ef4444; font-weight: 700; }
    </style>
    """,
    unsafe_allow_html=True,
)
# endregion ------------------------------------------------


# region -------------------- HTTP Helpers --------------------
def _get(path: str, **kwargs) -> httpx.Response:
    """Sends a GET request to the backend API."""
    return httpx.get(f"{API_BASE}{path}", timeout=REQUEST_TIMEOUT, **kwargs)


def _post(path: str, **kwargs) -> httpx.Response:
    """Sends a POST request to the backend API."""
    return httpx.post(f"{API_BASE}{path}", timeout=REQUEST_TIMEOUT, **kwargs)


def check_backend_health() -> bool:
    """Check if backend server is reachable and healthy."""
    try:
        r = httpx.get(f"{BACKEND_URL}/health", timeout=5.0)
        return r.status_code == 200
    except httpx.RequestError:
        return False


# endregion --------------------------------------------------


# region -------------------- Session State --------------------
def _init_state():
    """Initializes the Streamlit session state with default values."""
    defaults = {
        "page": "upload",
        "selected_doc_id": None,
        "documents": [],
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


_init_state()
# endregion ----------------------------------------------------


# region -------------------- Data Fetching --------------------
def fetch_documents() -> list[dict]:
    """Fetches the list of all documents from the backend."""
    try:
        r = _get("/documents/")
        r.raise_for_status()
        return r.json()
    except Exception:
        return []


def fetch_status(doc_id: int) -> dict | None:
    """Fetches the processing status of a specific document."""
    try:
        r = _get(f"/documents/{doc_id}/status")
        r.raise_for_status()
        return r.json()
    except Exception:
        return None


def upload_file(file) -> dict | None:
    """Uploads a PDF file to the backend for processing."""
    try:
        content_type = getattr(file, "type", None) or "application/octet-stream"
        r = httpx.post(
            f"{API_BASE}/documents/upload",
            files={"file": (file.name, file.getvalue(), content_type)},
            timeout=REQUEST_TIMEOUT,
        )
        r.raise_for_status()
        return r.json()
    except httpx.HTTPStatusError as exc:
        detail = exc.response.json().get("detail", str(exc))
        st.error(f"Upload failed: {detail}")
        return None
    except Exception as exc:
        st.error(f"Upload failed: {exc}")
        return None


# endregion ----------------------------------------------------


# region -------------------- Sidebar --------------------
def render_sidebar():
    """Renders the application's sidebar for navigation and document selection."""
    with st.sidebar:
        st.title("DocRAG")
        st.caption("RAG-based PDF Summarizer + QnA Generator")

        if check_backend_health():
            st.success("Backend connected")
        else:
            st.error("Backend disconnected")
            return

        st.divider()

        st.subheader("Navigation")
        pages = {
            "upload": "Upload Document",
        }
        for key, label in pages.items():
            if st.button(label, key=f"nav_{key}", use_container_width=True):
                st.session_state.page = key
                st.rerun()


# endregion -----------------------------------------------


# region -------------------- Page: Upload --------------------
def page_upload():
    """Renders the document upload page."""
    st.header("Upload Document")
    st.write(
        "Upload a PDF file to get started. The backend will extract, chunk "
        "and embed the text automatically."
    )

    uploaded = st.file_uploader(
        "Choose a PDF file",
        type=["pdf"],
        help="Maximum file size depends on server configuration (default 10 MB).",
    )

    if uploaded is not None:
        col1, col2 = st.columns([1, 2])
        with col1:
            st.metric("File", uploaded.name)
        with col2:
            size_mb = len(uploaded.getvalue()) / (1024 * 1024)
            st.metric("Size", f"{size_mb:.2f} MB")

        if st.button("Upload & Process", type="primary", use_container_width=True):
            with st.spinner("Uploading…"):
                result = upload_file(uploaded)

            if result:
                st.success(f"Uploaded successfully! Document ID: **{result['id']}**")
                st.session_state.selected_doc_id = result["id"]
                st.session_state.page = "document"

                # Poll until processing finishes (with a cap)
                status_placeholder = st.empty()
                progress = st.progress(0, text="Processing document…")
                max_polls = 120
                for i in range(max_polls):
                    status = fetch_status(result["id"])
                    if status is None:
                        break
                    current = status["status"]
                    status_placeholder.info(f"Status: **{current}**")
                    progress.progress(
                        min((i + 1) / max_polls, 0.95),
                        text="Processing…",
                    )
                    if current in ("completed", "failed"):
                        break
                    time.sleep(2)

                progress.progress(1.0, text="Done")
                if status and status["status"] == "completed":
                    st.success("Document processed successfully!")
                elif status and status["status"] == "failed":
                    st.error(
                        f"Processing failed: {status.get('error_message', 'Unknown error')}"
                    )

                time.sleep(1)
                st.rerun()


# endregion --------------------------------------------------


# region -------------------- Main --------------------
def main():
    """Main application entry point."""
    render_sidebar()

    page_upload()


if __name__ == "__main__":
    main()
# endregion --------------------------------------------
