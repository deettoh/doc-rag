"""DocRAG Streamlit Frontend - Main Application."""

import os

import httpx
import streamlit as st

# Configuration
BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")

st.set_page_config(
    page_title="DocRAG",
    layout="wide",
)


def check_backend_health() -> bool:
    """Check if backend is healthy."""
    try:
        response = httpx.get(f"{BACKEND_URL}/health", timeout=5.0)
        return response.status_code == 200
    except httpx.RequestError:
        return False


def main():
    """Main application entry point."""
    st.title("DocRAG")
    st.subheader("RAG-based PDF Summarizer + QnA Generator")

    # Backend health status
    with st.sidebar:
        st.header("System Status")
        if check_backend_health():
            st.success("Backend: Connected")
        else:
            st.error("Backend: Disconnected")

    # Main content placeholder
    st.markdown("---")
    st.info("**Unfinished**\n\nUpload functionality and document processing features ")

    # Placeholder sections for future features
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### Upload Document")
        st.file_uploader(
            "Choose a PDF file",
            type=["pdf"],
            disabled=True,
            help="Unfinished",
        )

    with col2:
        st.markdown("### Document Status")
        st.info("No documents uploaded yet.")


if __name__ == "__main__":
    main()
