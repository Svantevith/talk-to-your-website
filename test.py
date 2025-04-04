import streamlit as st
import threading
import time
import os
import psutil

TIMEOUT_SECONDS = 5  # Check if Streamlit is alive every 5 seconds


def monitor_app():
    """Continuously check if the Streamlit process is running."""
    pid = os.getpid()  # Get Streamlit process ID
    process = psutil.Process(pid)

    while True:
        if not process.is_running():  # Check if process is still alive
            st.session_state.app_closed = True
            st.experimental_rerun()  # Force UI update
            break  # Exit thread

        time.sleep(TIMEOUT_SECONDS)  # Check every 5 seconds


# Initialize session variables
if "app_closed" not in st.session_state:
    st.session_state.app_closed = False

# Start monitoring thread (only once)
if "monitor_started" not in st.session_state:
    st.session_state.monitor_started = True
    threading.Thread(target=monitor_app, daemon=True).start()

# Streamlit UI
st.title("Streamlit Self-Monitoring App")

if st.session_state.app_closed:
    st.error("⚠️ The app was closed or crashed!")
else:
    st.success("✅ The app is running normally.")
    st.write(f"Process ID: {os.getpid()}")
