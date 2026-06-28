import re
import youtube_transcript_api
import yt_dlp
from pathlib import Path
from typing import List, Dict, Any
from faster_whisper import WhisperModel
from app.config import settings

class TranscriptSourceSelector:
    def __init__(self):
        pass

    def extract_video_id(self, url: str) -> str:
        """
        Extracts the unique 11-character YouTube video ID using regular expressions.
        """
        pattern = r'(?:https?:\/\/)?(?:www\.)?(?:youtube\.com\/(?:[^\/\n\s]+\/\S+\/|(?:v|e(?:mbed)?)\/|\S*?[?&]v=)|youtu\.be\/)([a-zA-Z0-9_-]{11})'
        match = re.search(pattern, url)
        if not match:
            raise ValueError("Invalid YouTube URL provided.")
        return match.group(1)

    def _download_audio_fallback(self, video_url: str, video_id: str) -> Path:
        """
        Internal helper to extract low-bitrate compressed audio stream from YouTube 
        when native API captions are blocked or unavailable.
        """
        output_path = settings.AUDIO_CACHE_DIR / f"{video_id}.mp3"
        
        if output_path.exists():
            print(f"[Cache Hit] Reusing existing audio file: {output_path.name}")
            return output_path

        ydl_opts = {
            'format': 'bestaudio/best',
            'outtmpl': str(settings.AUDIO_CACHE_DIR / f"{video_id}.%(ext)s"),
            'postprocessors': [{
                'key': 'FFmpegExtractAudio',
                'preferredcodec': 'mp3',
                'preferredquality': '128',
            }],
            'quiet': True,
            'no_warnings': True,
        }

        print(f"[Network] Downloading compressed stream for fallback processing: {video_id}")
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([video_url])
            
        return output_path

    def get_raw_transcript(self, video_url: str, video_id: str = None) -> List[Dict[str, Any]]:
        """
        Main entrypoint for the Transcript Layer. Resolves native subtitles from YouTube 
        or processes a local Whisper ASR instance if tracking fails.
        """
        if not video_id:
            video_id = self.extract_video_id(video_url)
        
        try:
            print(f"[API Call] Attempting to pull official captions for ID: {video_id}")
            ytt_api = youtube_transcript_api.YouTubeTranscriptApi()
            fetched_transcript = ytt_api.fetch(video_id, languages=['en', 'vi'])
            raw_data = fetched_transcript.to_raw_data()
            
            print(f"[Success] Extracted native captions from YouTube API for ID: {video_id}")
            return [{
                "text": item["text"],
                "start": float(item["start"]),
                "duration": float(item["duration"]),
                "source_type": "youtube_caption"
            } for item in raw_data]
            
        # Fallback Pipeline: Local offline execution using localized Whisper weights
        except Exception as error_msg:
            print(f"[Warning] Native caption API unavailable ({error_msg}). Booting Whisper Fallback...")
            
            audio_file = self._download_audio_fallback(video_url, video_id)
            
            print("[Inference] Starting Faster Whisper runtime base model...")
            asr_model = WhisperModel("base", device="cpu", compute_type="int8")
            segments, info = asr_model.transcribe(str(audio_file), beam_size=5)
            
            whisper_output = []
            for segment in segments:
                whisper_output.append({
                    "text": segment.text.strip(),
                    "start": float(segment.start),
                    "duration": float(segment.end - segment.start),
                    "source_type": "whisper_fallback"
                })
                
            print(f"[Ingestion] Offline audio extraction finished for ID: {video_id}")
            return whisper_output