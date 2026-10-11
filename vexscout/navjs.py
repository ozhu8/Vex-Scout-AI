"""Closes the sidebar on phones after a menu choice, so the menu behaves like a
slide-out drawer (Streamlit has no Python API for collapsing it)."""
import streamlit as st

JS = """
export default function(component) {
  if (window.innerWidth > 700) return;
  const sidebar = document.querySelector('[data-testid="stSidebar"]');
  if (!sidebar || sidebar.getAttribute('aria-expanded') !== 'true') return;
  const close = document.querySelector('[data-testid="stSidebarCollapseButton"] button')
             || document.querySelector('[data-testid="stSidebarCollapseButton"]');
  if (close) close.click();
}
"""


def _register():
    return st.components.v2.component("vexscout_close_sidebar", js=JS)


_close = _register()


def close_sidebar(key):
    """Render once (with a fresh key) to close the phone sidebar."""
    global _close
    try:
        _close(key=key)
    except st.errors.StreamlitAPIException:  # registry reset (new runtime): register again
        _close = _register()
        _close(key=key)
