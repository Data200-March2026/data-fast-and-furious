"""
Logger — thin wrapper that maps classic logging calls to Streamlit UI.
"""

import streamlit as st


class Logger:
    """Maps Python logging conventions to Streamlit UI widgets."""

    def info(self, msg: str) -> None:
        st.toast(msg, icon="ℹ️")

    def success(self, msg: str) -> None:
        st.toast(msg, icon="✅")

    def error(self, msg: str) -> None:
        st.error(msg)

    def warning(self, msg: str) -> None:
        st.warning(msg)

    def section(self, msg: str) -> None:
        st.subheader(msg)


# Module-level singleton
logger = Logger()
