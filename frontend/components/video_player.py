import streamlit as st

def render_video_player(video_url: str, start_time: int = 0):
    """
    Renders the central YouTube interactive media layer with precise timestamp seek capacity.
    """
    st.markdown("### 📺 Video Player")
    if not video_url:
        st.info("Provide a valid YouTube resource URL to initialize the media viewport.")
        return

    st.video(video_url, start_time=start_time)
    st.caption(f"⏱️ Media synchronization anchor: **{start_time // 60:02d}:{start_time % 60:02d}** ({start_time}s)") 