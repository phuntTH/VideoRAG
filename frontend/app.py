import streamlit as st
import re
import requests
import time
from components.video_player import render_video_player
from components.preview_card import render_transcript_viewer
from components.search_interface import render_chat_interface

st.set_page_config(
    page_title="Next-Gen Video RAG Portal",
    layout="wide",
    initial_sidebar_state="expanded"
)

BACKEND_API_BASE = "http://127.0.0.1:8000/api"

# -----------------------------------------------------------------------------
# HIGH PERFORMANCE LOCAL DATA CACHING LAYER
# -----------------------------------------------------------------------------
@st.cache_data(ttl=1800, show_spinner=False)
def fetch_transcript_from_server_cached(video_id: str) -> list:
    """Fetches full analytical transcripts from the database with active local cache."""
    try:
        response = requests.get(f"{BACKEND_API_BASE}/video/transcript", params={"video_id": video_id}, timeout=None)
        if response.status_code == 200:
            return response.json().get("transcript", [])
    except Exception:
        pass
    return []

@st.cache_data(ttl=1800, show_spinner=False)
def fetch_gemini_suggestions_cached(video_id: str) -> list:
    """Fetches dynamically generated LLM context queries based on specific transcript metadata."""
    try:
        response = requests.get(f"{BACKEND_API_BASE}/video/suggestions", params={"video_id": video_id}, timeout=None)
        if response.status_code == 200:
            return response.json().get("suggested_questions", [])
    except Exception:
        pass
    return []

# -----------------------------------------------------------------------------
# CONTEXTUAL SESSION STATE INITIALIZATION
# -----------------------------------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []
if "history" not in st.session_state:
    st.session_state.history = []
if "indexed_videos" not in st.session_state:
    st.session_state.indexed_videos = {}  # Format: {video_id: cached_transcript_list}
if "current_start_time" not in st.session_state:
    st.session_state.current_start_time = 0
if "current_video_id" not in st.session_state:
    st.session_state.current_video_id = ""
if "active_query_trigger" not in st.session_state:
    st.session_state.active_query_trigger = None

# Non-blocking async-style polling engine flags
if "polling_active" not in st.session_state:
    st.session_state.polling_active = False
if "pipeline_status_string" not in st.session_state:
    st.session_state.pipeline_status_string = "idle"

# -----------------------------------------------------------------------------
# SIDEBAR CONTROL WORKSPACE
# -----------------------------------------------------------------------------
with st.sidebar:
    st.title("🛡️ Admin Control Deck")
    
    st.markdown("### 📚 Session Active Library")
    if st.session_state.indexed_videos:
        chosen_id = st.selectbox("Active Video Context Target:", options=list(st.session_state.indexed_videos.keys()))
        if chosen_id != st.session_state.current_video_id:
            st.session_state.current_video_id = chosen_id
            st.rerun()
    else:
        st.caption("No processed assets tracked inside the active context.")
        
    st.markdown("---")
    st.markdown("### 📜 System Global Queries")
    for past_query in reversed(st.session_state.history[-5:]):
        if st.button(f"🔍 {past_query}", key=f"sidebar_hist_{past_query}", use_container_width=True):
            st.session_state.active_query_trigger = past_query
            st.rerun()

# -----------------------------------------------------------------------------
# APPLICATION VIEWPORT COMPONENT MATRIX
# -----------------------------------------------------------------------------
st.title("⚡ Enterprise Multimodal Video Hybrid RAG Engine")
st.markdown("Abstract highly contextualized insights from video data assets utilizing FAISS dense indexing and BM25 sparse matching.")

# Instantiate main structural layout
left_viewport, right_viewport = st.columns([1, 1], gap="large")

with left_viewport:
    st.markdown("### 🎬 Data Ingestion Engine")
    input_url = st.text_input("YouTube Context Source URL:", value="")
    
    video_id_regex = r'(?:https?:\/\/)?(?:www\.)?(?:youtube\.com\/(?:[^\/\n\s]+\/\S+\/|(?:v|e(?:mbed)?)\/|\S*?[?&]v=)|youtu\.be\/)([a-zA-Z0-9_-]{11})'
    match_object = re.search(video_id_regex, input_url)
    extracted_id = match_object.group(1) if match_object else ""
    
    if extracted_id and extracted_id != st.session_state.current_video_id and not st.session_state.polling_active:
        st.session_state.current_video_id = extracted_id
        st.session_state.pipeline_status_string = "idle"
        
        # Auto-recover transcript if already processed by backend
        try:
            status_res = requests.get(f"{BACKEND_API_BASE}/video/status", params={"video_id": extracted_id}, timeout=2)
            if status_res.status_code == 200 and status_res.json().get("status") == "completed":
                st.session_state.indexed_videos[extracted_id] = fetch_transcript_from_server_cached(extracted_id)
        except Exception:
            pass

    if st.button("⚙️ Initialize Multimodal Ingestion Pipeline", use_container_width=True):
        if extracted_id:
            try:
                init_req = requests.post(f"{BACKEND_API_BASE}/video/process", json={"url": input_url, "video_id": extracted_id}, timeout=60)
                if init_req.status_code in [200, 202]:
                    # Activate structural non-blocking polling sequence out of button handler scope
                    st.session_state.polling_active = True
                    st.session_state.pipeline_status_string = "queued"
                    st.toast("Asynchronous data ingestion job enqueued!", icon="🚀")
                    st.rerun()
                else:
                    st.error("Server ingestion allocation failure.")
            except Exception as e:
                st.error(f"Fatal handshake interface failure during initiation: {e}")

    st.markdown("---")
    
    render_video_player(video_url=input_url, start_time=st.session_state.current_start_time)
    
    active_transcript_source = st.session_state.indexed_videos.get(st.session_state.current_video_id, [])
    render_transcript_viewer(active_transcript_source, st.session_state.current_start_time)

with right_viewport:
    if st.session_state.current_video_id in st.session_state.indexed_videos:
        fetched_suggestions = fetch_gemini_suggestions_cached(st.session_state.current_video_id)
        if fetched_suggestions:
            st.markdown("💡 **LLM Contextual Query Suggestions:**")
            chip_columns = st.columns(len(fetched_suggestions[:3]))
            for index, prompt_suggestion in enumerate(fetched_suggestions[:3]):
                if chip_columns[index].button(prompt_suggestion, key=f"chip_link_{index}", use_container_width=True):
                    st.session_state.active_query_trigger = prompt_suggestion
                    st.rerun()
                    
    st.markdown("---")
    
    render_chat_interface(backend_url=BACKEND_API_BASE)

if st.session_state.polling_active:
    status_toast_container = st.empty()
    status_toast_container.warning(f"🔄 Processing Video: Operational pipeline current phase is **{st.session_state.pipeline_status_string.upper()}**")
    
    try:
        poll_check = requests.get(f"{BACKEND_API_BASE}/video/status", params={"video_id": st.session_state.current_video_id}, timeout=2)
        if poll_check.status_code == 200:
            server_reported_phase = poll_check.json().get("status", "queued")
            st.session_state.pipeline_status_string = server_reported_phase
            
            if server_reported_phase == "completed":
                st.session_state.polling_active = False
                st.session_state.indexed_videos[st.session_state.current_video_id] = fetch_transcript_from_server_cached(st.session_state.current_video_id)
                status_toast_container.empty()
                st.toast("Target vector partitions compiled successfully!", icon="✅")
                st.rerun()
                
            elif server_reported_phase == "failed":
                st.session_state.polling_active = False
                status_toast_container.empty()
                st.error("Server background asynchronous operation failed.")
                st.rerun()
    except Exception:
        pass  
    time.sleep(2.0)
    st.rerun()