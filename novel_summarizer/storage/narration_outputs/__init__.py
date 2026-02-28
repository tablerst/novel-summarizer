from novel_summarizer.storage.narration_outputs.base import NarrationOutput
from novel_summarizer.storage.narration_outputs.crud import (
    get_latest_narration_output_for_chapter,
    get_narration_output,
    upsert_narration_output,
)

__all__ = [
    "NarrationOutput",
    "get_narration_output",
    "get_latest_narration_output_for_chapter",
    "upsert_narration_output",
]
