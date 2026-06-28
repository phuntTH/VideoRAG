import streamlit as st
import re
from typing import List, Dict, Any

def highlight_query_keywords(text: str, query: str) -> str:
    """Highlights relevant search queries using standard Markdown bold markers."""
    if not query:
        return text
    words = query.strip().split()
    for word in words:
        if len(word) > 3:  # Skip trivial short grammatical terms
            compiled = re.compile(re.escape(word), re.IGNORECASE)
            text = compiled.sub(f"**{word}**", text)
    return text

def render_evidence_card(card_id: str, preview_text: str, timestamp: int, score: float, origin_query: str):
    """
    Renders a standalone grounded citation card (Perplexity style) with Reranker scores.
    """
    confidence_percentage = int(score * 100)
    timestamp_int = int(timestamp)
    time_str = f"{timestamp_int // 60:02d}:{timestamp_int % 60:02d}"
    
    processed_text = highlight_query_keywords(preview_text, origin_query)
    
    col_content, col_action = st.columns([4, 1])
    with col_content:
        st.markdown(f"**[{time_str}]** {processed_text}")
        st.caption(f"🎯 Match Confidence: `{confidence_percentage}%` via BGE-Reranker")
    with col_action:
        if st.button("▶ Jump", key=f"citation_jump_{card_id}_{timestamp}", use_container_width=True):
            st.session_state.current_start_time = int(timestamp)
            st.rerun()

def render_transcript_viewer(transcript_data: List[Dict[str, Any]], current_start_time: int):
    """
    Renders an enterprise-grade transcript browser utilizing row pagination, 
    sub-string filtering, and state-synchronized highlight mapping.
    """
    st.markdown("### 📜 Transcript Text Browser")
    
    if not transcript_data:
        st.info("Transcript space is empty. Trigger video indexing on the ingestion module.")
        return

    filter_query = st.text_input(
        "🔍 Filter transcript sentences:", 
        placeholder="Search keywords inside transcript (e.g., BM25, Attention)...",
        key="internal_transcript_search_input"
    )
    
    filtered_list = transcript_data
    if filter_query.strip():
        filtered_list = [
            line for line in transcript_data 
            if filter_query.lower() in line.get("text", "").lower()
        ]

    active_global_idx = -1
    if current_start_time > 0 and filtered_list:
        lowest_delta = float('inf')
        for index, element in enumerate(filtered_list):
            delta = abs(int(element.get("start", element.get("start_time", 0))) - current_start_time)
            if delta < lowest_delta:
                lowest_delta = delta
                active_global_idx = index

    rows_per_page = 20
    total_rows = len(filtered_list)
    calculated_pages = max(1, (total_rows + rows_per_page - 1) // rows_per_page)
    computed_target_page = 1
    if active_global_idx != -1:
        computed_target_page = (active_global_idx // rows_per_page) + 1
    computed_target_page = min(computed_target_page, calculated_pages)

    col_meta, col_control = st.columns([3, 1])
    with col_meta:
        st.caption(f"Rendered viewport: showing {min(rows_per_page, total_rows)} of {total_rows} matching records.")
    with col_control:
        selected_page = st.number_input(
            "Page Selector", min_value=1, max_value=calculated_pages, 
            value=computed_target_page, step=1, key="dom_page_stepper"
        )

    offset_start = (selected_page - 1) * rows_per_page
    offset_end = min(offset_start + rows_per_page, total_rows)
    rendered_view_slice = filtered_list[offset_start:offset_end]
    with st.container(height=360, border=True):
        for local_idx, item in enumerate(rendered_view_slice):
            actual_global_idx = offset_start + local_idx
            sentence_text = item.get("text", "")
            seconds_anchor = int(item.get("start", item.get("start_time", 0)))
            
            is_row_active = (actual_global_idx == active_global_idx)
            bg_style = "rgba(255, 217, 0, 0.22)" if is_row_active else "transparent"
            border_style = "3px solid #FFD700" if is_row_active else "none"
            
            col_nav, col_phrase = st.columns([1, 5])
            with col_nav:
                time_lbl = f"{seconds_anchor // 60:02d}:{seconds_anchor % 60:02d}"
                if st.button(time_lbl, key=f"line_btn_{actual_global_idx}_{seconds_anchor}", use_container_width=True):
                    st.session_state.current_start_time = seconds_anchor
                    st.rerun()
            with col_phrase:
                st.markdown(
                    f"<div style='background-color: {bg_style}; border-left: {border_left_css_calc(border_style)}; padding: 3px 8px; border-radius: 4px; font-size: 13.5px;'>"
                    f"{sentence_text}"
                    f"</div>", 
                    unsafe_allow_html=True
                )

def border_left_css_calc(val: str) -> str:
    return "3px solid #FFD700" if "3px" in val else "none"