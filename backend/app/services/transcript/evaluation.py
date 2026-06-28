import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))


import re
from typing import List, Dict, Any
from app.core.constants import NOISE_PATTERNS

def evaluate_transcript_quality(transcript: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Evaluates the quality of incoming transcripts (0-100).
    Triggers mandatory Whisper fallback if Quality Score < 75.
    """
    if not transcript:
        return {"quality_score": 0.0, "requires_whisper": True, "reason": "Empty transcript detected."}

    total_words = 0
    total_duration_minutes = 0.0
    noise_blocks = 0
    total_blocks = len(transcript)

    noise_regex = re.compile('|'.join(NOISE_PATTERNS))

    for block in transcript:
        text = block.get("text", "").strip()
        duration = float(block.get("duration", 0.0))

        if noise_regex.search(text) or text == "":
            noise_blocks += 1

        cleaned_text = noise_regex.sub('', text).strip()
        words_count = len(cleaned_text.split()) if cleaned_text else 0
        total_words += words_count
        total_duration_minutes += (duration / 60.0)

    if total_duration_minutes <= 0:
        return {"quality_score": 0.0, "requires_whisper": True, "reason": "Zero speech duration detected."}

    wpm = total_words / total_duration_minutes

    optimal_wpm = 145.0
    velocity_score = 100.0 - (0.5 * abs(wpm - optimal_wpm))
    velocity_score = max(0.0, min(100.0, velocity_score))
    noise_penalty_percentage = (noise_blocks / total_blocks) * 100.0

    final_score = round(max(0.0, velocity_score - noise_penalty_percentage), 2)
    
    requires_whisper = final_score < 75.0

    return {
        "quality_score": final_score,
        "wpm": round(wpm, 2),
        "noise_block_ratio": round(noise_blocks / total_blocks, 2),
        "requires_whisper": requires_whisper,
        "reason": f"Score {final_score}/100 based on WPM={round(wpm, 1)} and Noise Ratio={round(noise_blocks/total_blocks, 2)}"
    }