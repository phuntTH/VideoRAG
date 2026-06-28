import streamlit as st
import requests
import time
from components.preview_card import render_evidence_card

def render_chat_interface(backend_url: str):
    """
    Manages the conversational loop workflow, rendering asynchronous chat components 
    and integrating granular grounding cards.
    """
    st.markdown("### 💬 Conversational RAG Inference Hub")
    
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message["role"] == "assistant" and "grounding_data" in message:
                g_data = message["grounding_data"]
                
                with st.expander("🔗 Grounded Source Citations (Verified)", expanded=True):
                    previews = g_data.get("preview", [])
                    timestamps = g_data.get("timestamps", [])
                    scores = g_data.get("rerank_scores", [])
                    
                    for idx, (snippet, t_stamp) in enumerate(zip(previews, timestamps)):
                        match_score = scores[idx] if idx < len(scores) else 0.85
                        render_evidence_card(
                            card_id=f"msg_{message['id']}_{idx}",
                            preview_text=snippet,
                            timestamp=t_stamp,
                            score=match_score,
                            origin_query=message.get("query_origin", "")
                        )
    user_input = st.chat_input("Query any knowledge block encapsulated inside the active video library...")
    if st.session_state.active_query_trigger:
        user_input = st.session_state.active_query_trigger
        st.session_state.active_query_trigger = None

    if user_input:
        if st.session_state.current_video_id not in st.session_state.indexed_videos:
            st.warning("⚠️ Ingestion validation failed: You must index the target video sequence prior to executing a query loop.")
            return
        st.session_state.messages.append({"role": "user", "content": user_input})
        if user_input not in st.session_state.history:
            st.session_state.history.append(user_input)
        with st.spinner("Executing dense hybrid vector search & reranking..."):
            try:
                payload = {
                    "video_id": st.session_state.current_video_id,
                    "query": user_input
                }
                res = requests.post(f"{backend_url}/search", json=payload, timeout=30)
                
                if res.status_code == 200:
                    api_data = res.json()
                    st.session_state.messages.append({
                        "id": int(time.time()),
                        "role": "assistant",
                        "content": api_data.get("answer"),
                        "query_origin": user_input,
                        "grounding_data": api_data
                    })
                else:
                    st.error(f"Inference Failure: Backend api returned error status {res.status_code}")
            except Exception as e:
                st.error(f"Critical execution error establishing backend pipeline handshakes: {e}")
                
        st.rerun()