import os
import tempfile
import torch

from dotenv import load_dotenv
from fastapi import FastAPI, UploadFile, File
from faster_whisper import WhisperModel
from pyannote.audio import Pipeline
from pyannote.core import Segment

load_dotenv()

app = FastAPI()

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
COMPUTE_TYPE = "float16" if DEVICE == "cuda" else "int8"

HF_TOKEN = os.getenv("HF_TOKEN")

WHISPER_MODEL = "./models/ggml-large-v3-turbo"
PYANNOTE_MODEL = "pyannote/speaker-diarization-community-1"


# Load model 1 lần khi server start
whisper_model = WhisperModel(
    WHISPER_MODEL,
    device=DEVICE,
    compute_type=COMPUTE_TYPE
)

diarization_pipeline = Pipeline.from_pretrained(
    PYANNOTE_MODEL,
    token=HF_TOKEN
)

if DEVICE == "cuda":
    diarization_pipeline.to(torch.device("cuda"))


def find_speaker(diarization, start, end):
    if start is None or end is None:
        return "UNKNOWN"

    if start == end:
        end += 0.05

    cropped = diarization.crop(
        Segment(start, end)
    )

    speakers = cropped.labels()

    if not speakers:
        return "UNKNOWN"

    return max(
        speakers,
        key=lambda s: cropped.label_duration(s)
    )


def merge_words(words):
    if not words:
        return []

    result = []

    current = {
        "start": words[0]["start"],
        "end": words[0]["end"],
        "speaker": words[0]["speaker"],
        "text": words[0]["text"]
    }

    for word in words[1:]:
        if word["speaker"] == current["speaker"]:
            current["text"] += word["text"]
            current["end"] = word["end"]
        else:
            current["text"] = current["text"].strip()
            result.append(current)

            current = {
                "start": word["start"],
                "end": word["end"],
                "speaker": word["speaker"],
                "text": word["text"]
            }

    current["text"] = current["text"].strip()
    result.append(current)

    return result


@app.get("/")
def root():
    return {
        "status": "ok",
        "device": DEVICE
    }


@app.post("/transcribe")
async def transcribe(
    file: UploadFile = File(...)
):
    suffix = os.path.splitext(file.filename)[1]

    with tempfile.NamedTemporaryFile(
        delete=False,
        suffix=suffix
    ) as temp:
        temp.write(await file.read())
        audio_path = temp.name

    try:
        # Whisper
        segments, info = whisper_model.transcribe(
            audio_path,
            language="ja",
            beam_size=5,
            word_timestamps=True,
            vad_filter=True
        )

        whisper_segments = []

        for segment in segments:
            whisper_segments.append({
                "start": segment.start,
                "end": segment.end,
                "text": segment.text.strip(),
                "words": [
                    {
                        "start": word.start,
                        "end": word.end,
                        "text": word.word
                    }
                    for word in segment.words
                    if word.start is not None
                    and word.end is not None
                ]
            })

        # Pyannote
        output = diarization_pipeline(
            audio_path
        )

        diarization = output.speaker_diarization

        # Assign speaker từng word
        words = []

        for segment in whisper_segments:
            for word in segment["words"]:
                speaker = find_speaker(
                    diarization,
                    word["start"],
                    word["end"]
                )

                words.append({
                    "start": word["start"],
                    "end": word["end"],
                    "speaker": speaker,
                    "text": word["text"]
                })

        result = merge_words(words)

        return {
            "language": info.language,
            "duration": info.duration,
            "segments": result
        }

    finally:
        if os.path.exists(audio_path):
            os.remove(audio_path)