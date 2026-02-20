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


def generate_summary(doc_id: int) -> dict | None:
    """Triggers the generation of a document summary on the backend."""
    try:
        r = _post(f"/documents/{doc_id}/summarize")
        r.raise_for_status()
        return r.json()
    except httpx.HTTPStatusError as exc:
        detail = exc.response.json().get("detail", str(exc))
        st.error(f"Summarization failed: {detail}")
        return None
    except Exception as exc:
        st.error(f"Summarization failed: {exc}")
        return None


def generate_questions(doc_id: int, num: int = 5) -> dict | None:
    """Triggers the generation of study questions for a document."""
    try:
        r = _post(
            f"/documents/{doc_id}/questions",
            json={"num_questions": num},
        )
        r.raise_for_status()
        return r.json()
    except httpx.HTTPStatusError as exc:
        detail = exc.response.json().get("detail", str(exc))
        st.error(f"Question generation failed: {detail}")
        return None
    except Exception as exc:
        st.error(f"Question generation failed: {exc}")
        return None


def submit_answer(doc_id: int, question_id: int, answer_text: str) -> dict | None:
    """Submits a user's answer for evaluation against a specific question."""
    try:
        r = _post(
            f"/documents/{doc_id}/questions/{question_id}/answer",
            json={"user_answer": answer_text},
        )
        r.raise_for_status()
        return r.json()
    except httpx.HTTPStatusError as exc:
        detail = exc.response.json().get("detail", str(exc))
        st.error(f"Answer submission failed: {detail}")
        return None
    except Exception as exc:
        st.error(f"Answer submission failed: {exc}")
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
            "document": "Document View",
        }
        for key, label in pages.items():
            if st.button(label, key=f"nav_{key}", use_container_width=True):
                st.session_state.page = key
                st.rerun()

        st.divider()

        st.subheader("Documents")
        docs = fetch_documents()
        st.session_state.documents = docs

        if not docs:
            st.info("No documents uploaded yet.")
            return

        doc_options = {d["id"]: f"{d['filename']}" for d in docs}
        selected = st.selectbox(
            "Select Document",
            options=list(doc_options.keys()),
            format_func=lambda x: doc_options[x],
            index=(
                list(doc_options.keys()).index(st.session_state.selected_doc_id)
                if st.session_state.selected_doc_id in doc_options
                else 0
            ),
        )
        if selected != st.session_state.selected_doc_id:
            st.session_state.selected_doc_id = selected
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


# region -------------------- Page: Document View --------------------
def page_document():
    """Renders the document details view including status, summary, and QnA."""
    doc_id = st.session_state.selected_doc_id
    if doc_id is None:
        st.info("Select a document from the sidebar, or upload one first.")
        return

    doc_meta = next((d for d in st.session_state.documents if d["id"] == doc_id), None)
    if doc_meta is None:
        st.warning("Document not found. Try refreshing.")
        return

    st.header(f"{doc_meta['filename']}")

    status_data = fetch_status(doc_id)
    if status_data is None:
        st.error("Could not fetch document status.")
        return

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric(
            "Status",
            f"{status_data['status']}",
        )
    with col2:
        pages = status_data.get("page_count")
        st.metric("Pages", pages if pages else "—")
    with col3:
        size_kb = doc_meta["file_size_bytes"] / 1024
        st.metric("Size", f"{size_kb:.1f} KB")

    if status_data["status"] == "processing":
        st.info("Document is still being processed. Please wait…")
        if st.button("Refresh Status"):
            st.rerun()
        return

    if status_data["status"] == "failed":
        st.error(
            f"Processing failed: {status_data.get('error_message', 'Unknown error')}"
        )
        return

    if status_data["status"] != "completed":
        st.warning(f"Unexpected status: {status_data['status']}")
        return

    st.divider()

    tab_summary, tab_questions = st.tabs(["Summary", "Questions & Answers"])

    with tab_summary:
        _render_summary_tab(doc_id)

    with tab_questions:
        _render_questions_tab(doc_id)


def _render_summary_tab(doc_id: int):
    """Internal helper to render the summary tab content."""
    st.subheader("Document Summary")
    if st.button("Generate Summary", type="primary", key="gen_summary"):
        with st.spinner("Generating summary… This may take a moment."):
            result = generate_summary(doc_id)
        if result:
            st.session_state[f"summary_{doc_id}"] = result
            st.rerun()

    summary = st.session_state.get(f"summary_{doc_id}")
    if summary:
        st.markdown(summary["content"])
        if summary.get("page_citations"):
            cited = ", ".join(str(p) for p in summary["page_citations"])
            st.caption(f"Pages cited: {cited}")
    else:
        st.info("Click **Generate Summary** to create a summary of this document.")


def _render_questions_tab(doc_id: int):
    """Internal helper to render the study questions tab content."""
    st.subheader("Study Questions")

    col1, col2 = st.columns([3, 1])
    with col1:
        num_q = st.slider(
            "Number of questions", min_value=1, max_value=20, value=5, key="num_q"
        )
    with col2:
        gen_btn = st.button("Generate", type="primary", key="gen_questions")

    if gen_btn:
        with st.spinner("Generating questions…"):
            result = generate_questions(doc_id, num_q)
        if result:
            st.session_state[f"questions_{doc_id}"] = result["questions"]
            st.rerun()

    questions = st.session_state.get(f"questions_{doc_id}", [])
    if not questions:
        st.info("Click **Generate** to create study questions from this document.")
        return

    for i, q in enumerate(questions):
        with st.expander(
            f"Q{i + 1}: {q['content'][:100]}{'…' if len(q['content']) > 100 else ''}",
            expanded=False,
        ):
            st.markdown(f"**Question:** {q['content']}")

            answer_key = f"answer_result_{doc_id}_{q['id']}"
            existing_answer = st.session_state.get(answer_key)

            if existing_answer:
                _display_answer_result(existing_answer)
            else:
                user_ans = st.text_area(
                    "Your answer:",
                    key=f"ans_input_{doc_id}_{q['id']}",
                    placeholder="Type your answer here…",
                )
                if st.button("Submit Answer", key=f"submit_{doc_id}_{q['id']}"):
                    if not user_ans.strip():
                        st.warning("Please type an answer first.")
                    else:
                        with st.spinner("Evaluating…"):
                            result = submit_answer(doc_id, q["id"], user_ans)
                        if result:
                            st.session_state[answer_key] = result
                            st.rerun()


def _display_answer_result(answer: dict):
    """Internal helper to display the score and feedback for a quiz answer."""
    score = answer.get("score")
    feedback = answer.get("feedback", "")

    col1, col2 = st.columns([1, 3])
    with col1:
        if score is not None:
            pct = int(score * 100)
            if pct >= 70:
                css_class = "score-high"
            elif pct >= 40:
                css_class = "score-mid"
            else:
                css_class = "score-low"
            st.markdown(
                f'<p class="{css_class}" style="font-size:2rem;">{pct}%</p>',
                unsafe_allow_html=True,
            )
        else:
            st.metric("Score", "N/A")

    with col2:
        st.markdown(f"**Your answer:** {answer['user_answer']}")
        if feedback:
            st.info(f"{feedback}")


# endregion -----------------------------------------------------------


# region -------------------- Main --------------------
def main():
    """Main application entry point."""
    render_sidebar()

    page = st.session_state.page
    if page == "upload":
        page_upload()
    elif page == "document":
        page_document()
    else:
        page_upload()


if __name__ == "__main__":
    main()
# endregion --------------------------------------------
