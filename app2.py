import os
import gc
import time
import logging
import tempfile
import subprocess
from enum import Enum

import torch
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from faster_whisper import WhisperModel
from nemo.collections.asr.models import SortformerEncLabelModel


# ============================================================
# LOGGING
# ============================================================

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("speech-api")


# ============================================================
# FASTAPI
# ============================================================

app = FastAPI(title="Whisper Speech API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"]
)


# ============================================================
# ENUM PARAM
# ============================================================

class WhisperModelName(str, Enum):
    small = "small"
    large_v3 = "large-v3"
    large_v3_turbo = "large-v3-turbo"


class LanguageName(str, Enum):
    auto = "auto"
    ja = "ja"
    en = "en"
    vi = "vi"


# ============================================================
# DEVICE
# ============================================================

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
COMPUTE_TYPE = "float16" if DEVICE == "cuda" else "int8"

logger.info("============================================")
logger.info("Speech API starting")
logger.info("Device       : %s", DEVICE)
logger.info("Compute type : %s", COMPUTE_TYPE)

if DEVICE == "cuda":
    logger.info("GPU          : %s", torch.cuda.get_device_name(0))
    logger.info("VRAM         : %.2f GB", torch.cuda.get_device_properties(0).total_memory / 1024**3)

logger.info("============================================")


# ============================================================
# NEMO SORTFORMER DIARIZATION
# ============================================================

SORTFORMER_MODEL = os.getenv("SORTFORMER_MODEL", "nvidia/diar_sortformer_4spk-v1")

logger.info("Loading NeMo Sortformer model...")
logger.info("Sortformer model: %s", SORTFORMER_MODEL)

sortformer_start = time.perf_counter()

# For a local .nemo checkpoint, set SORTFORMER_MODEL to its file path.
if os.path.isfile(SORTFORMER_MODEL) and SORTFORMER_MODEL.lower().endswith(".nemo"):
    diarization_model = SortformerEncLabelModel.restore_from(SORTFORMER_MODEL)
else:
    diarization_model = SortformerEncLabelModel.from_pretrained(SORTFORMER_MODEL)

diarization_model.eval()

if DEVICE == "cuda":
    diarization_model = diarization_model.to(torch.device("cuda"))

logger.info("Sortformer loaded in %.2f sec", time.perf_counter() - sortformer_start)


# ============================================================
# WHISPER MODEL
# ============================================================

whisper_model = None
current_whisper_model = None


def get_whisper_model(model_name: str):
    global whisper_model, current_whisper_model

    if whisper_model is not None and current_whisper_model == model_name:
        logger.info("Reuse Whisper model: %s", model_name)
        return whisper_model

    if whisper_model is not None:
        logger.info("Unload Whisper model: %s", current_whisper_model)
        del whisper_model
        whisper_model = None
        current_whisper_model = None
        gc.collect()

        if DEVICE == "cuda":
            torch.cuda.empty_cache()

    logger.info("Loading Whisper model: %s", model_name)
    started = time.perf_counter()

    whisper_model = WhisperModel(model_name, device=DEVICE, compute_type=COMPUTE_TYPE)
    current_whisper_model = model_name

    logger.info("Whisper model %s loaded in %.2f sec", model_name, time.perf_counter() - started)
    return whisper_model


# ============================================================
# AUDIO
# ============================================================

def convert_to_wav(input_path: str) -> str:
    fd, wav_path = tempfile.mkstemp(suffix=".wav")
    os.close(fd)

    try:
        subprocess.run(
            ["ffmpeg", "-y", "-i", input_path, "-vn", "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", wav_path],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE
        )
        return wav_path
    except subprocess.CalledProcessError as ex:
        safe_remove(wav_path)
        error = ex.stderr.decode("utf-8", errors="ignore") if ex.stderr else str(ex)
        raise RuntimeError(f"FFmpeg conversion failed: {error}")



def preprocess_audio(input_path: str) -> str:
    """
    Lightweight speech preprocessing using FFmpeg only:
    - highpass 80 Hz: remove low-frequency rumble/hum
    - afftdn: reduce steady background noise conservatively
    - dynaudnorm: raise quiet speech while limiting excessive gain

    No additional Python package is required.
    """
    fd, wav_path = tempfile.mkstemp(suffix=".wav")
    os.close(fd)

    audio_filter = (
        "highpass=f=80,"
        "afftdn=nr=10:nf=-35:tn=1,"
        "dynaudnorm=f=150:g=7:p=0.95:m=4"
    )

    try:
        subprocess.run(
            [
                "ffmpeg", "-y",
                "-i", input_path,
                "-vn",
                "-ac", "1",
                "-ar", "16000",
                "-af", audio_filter,
                "-c:a", "pcm_s16le",
                wav_path,
            ],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )
        return wav_path
    except subprocess.CalledProcessError as ex:
        safe_remove(wav_path)
        error = ex.stderr.decode("utf-8", errors="ignore") if ex.stderr else str(ex)
        raise RuntimeError(f"FFmpeg audio preprocessing failed: {error}")


def safe_remove(path: str):
    if not path or not os.path.exists(path):
        return

    gc.collect()

    for _ in range(5):
        try:
            os.remove(path)
            logger.info("Deleted temp file: %s", path)
            return
        except PermissionError:
            time.sleep(0.2)
        except Exception as ex:
            logger.warning("Could not delete temp file %s: %s", path, ex)
            return

    logger.warning("Could not delete temp file because it is still in use: %s", path)


# ============================================================
# SORTFORMER OUTPUT + SPEAKER MATCHING
# ============================================================

def normalize_speaker_label(raw_speaker) -> str:
    value = str(raw_speaker).strip()
    digits = "".join(ch for ch in value if ch.isdigit())

    if digits:
        return f"SPEAKER_{int(digits):02d}"

    value = value.upper().replace(" ", "_")
    return value if value.startswith("SPEAKER_") else f"SPEAKER_{value}"


def parse_sortformer_segments(predicted_segments):
    """
    Convert NeMo Sortformer output into:
      [{"start": float, "end": float, "speaker": "SPEAKER_00"}, ...]

    NeMo versions may return either:
    - list[list[str]] for batched audio, where each string is "start end speaker"
    - list[str]
    - objects exposing start/end/speaker attributes
    """
    if predicted_segments is None:
        return []

    raw = predicted_segments

    # diarize(audio=[single_file]) commonly returns one outer item for the file.
    if isinstance(raw, (list, tuple)) and len(raw) == 1 and isinstance(raw[0], (list, tuple)):
        raw = raw[0]

    parsed = []

    for item in raw:
        start = end = speaker = None

        if isinstance(item, str):
            parts = item.strip().split()
            if len(parts) >= 3:
                start, end, speaker = parts[0], parts[1], parts[2]
        elif isinstance(item, dict):
            start = item.get("start")
            end = item.get("end")
            speaker = item.get("speaker")
        else:
            start = getattr(item, "start", None)
            end = getattr(item, "end", None)
            speaker = getattr(item, "speaker", None)

        if start is None or end is None or speaker is None:
            logger.warning("Skip unknown Sortformer segment format: %r", item)
            continue

        parsed.append({
            "start": float(start),
            "end": float(end),
            "speaker": normalize_speaker_label(speaker)
        })

    parsed.sort(key=lambda x: (x["start"], x["end"], x["speaker"]))
    return parsed


def find_segment_speaker(diarization_segments, start, end):
    if start is None or end is None or end <= start:
        return "UNKNOWN"

    overlap_by_speaker = {}

    for diar in diarization_segments:
        overlap = max(0.0, min(end, diar["end"]) - max(start, diar["start"]))
        if overlap <= 0:
            continue

        speaker = diar["speaker"]
        overlap_by_speaker[speaker] = overlap_by_speaker.get(speaker, 0.0) + overlap

    if not overlap_by_speaker:
        return "UNKNOWN"

    return max(overlap_by_speaker, key=overlap_by_speaker.get)


# ============================================================
# HEALTH CHECK
# ============================================================

@app.get("/")
def root():
    return {
        "status": "ok",
        "device": DEVICE,
        "computeType": COMPUTE_TYPE,
        "currentWhisperModel": current_whisper_model,
        "diarizationModel": SORTFORMER_MODEL,
        "models": ["small", "large-v3", "large-v3-turbo"],
        "languages": ["auto", "ja", "en", "vi"]
    }


# ============================================================
# TRANSCRIBE
# ============================================================

@app.post("/transcribe")
async def transcribe(
    file: UploadFile = File(...),
    language: LanguageName = Form(LanguageName.ja),
    model: WhisperModelName = Form(WhisperModelName.small),
    num_speakers: int | None = Form(None)
):
    request_started = time.perf_counter()
    audio_path = None
    processed_audio_path = None

    language_value = language.value
    model_name = model.value

    if num_speakers is not None and not 1 <= num_speakers <= 4:
        raise HTTPException(status_code=400, detail="Sortformer supports 1 to 4 speakers")

    logger.info("")
    logger.info("============================================")
    logger.info("NEW TRANSCRIPTION REQUEST")
    logger.info("File     : %s", file.filename)
    logger.info("Model    : %s", model_name)
    logger.info("Language : %s", language_value)
    logger.info("Speakers : %s", num_speakers if num_speakers is not None else "auto")
    logger.info("Device   : %s", DEVICE)
    logger.info("============================================")

    try:
        # ====================================================
        # SAVE FILE
        # ====================================================

        suffix = os.path.splitext(file.filename)[1]
        data = await file.read()

        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp:
            temp.write(data)
            audio_path = temp.name

        logger.info("File size : %.2f MB", len(data) / 1024 / 1024)
        logger.info("Temp file : %s", audio_path)

        # ====================================================
        # AUDIO PREPROCESS
        # ====================================================

        logger.info("")
        logger.info("---------- PREPROCESS START --------")

        preprocess_started = time.perf_counter()
        processed_audio_path = preprocess_audio(audio_path)
        preprocess_elapsed = time.perf_counter() - preprocess_started

        logger.info("Processed audio : %s", processed_audio_path)
        logger.info("Filters         : highpass 80Hz + afftdn + dynaudnorm")
        logger.info("Format          : WAV PCM 16-bit / mono / 16 kHz")
        logger.info("Preprocess time : %.2f sec", preprocess_elapsed)
        logger.info("---------- PREPROCESS END ----------")

        # ====================================================
        # LOAD WHISPER
        # ====================================================

        model_instance = get_whisper_model(model_name)

        # ====================================================
        # LANGUAGE
        # ====================================================

        whisper_language = None if language_value == "auto" else language_value

        # ====================================================
        # WHISPER
        # ====================================================

        logger.info("")
        logger.info("---------- WHISPER START ----------")

        whisper_started = time.perf_counter()

        segments, info = model_instance.transcribe(
            processed_audio_path,
            language=whisper_language,
            beam_size=5,
            word_timestamps=True,
            vad_filter=True,
            vad_parameters=dict(
                threshold=0.3,
                min_speech_duration_ms=50,
                min_silence_duration_ms=500,
                speech_pad_ms=300
            )
        )

        whisper_segments = []
        word_count = 0

        for index, segment in enumerate(segments, start=1):
            segment_words = []

            if segment.words:
                for word in segment.words:
                    if word.start is None or word.end is None:
                        continue

                    segment_words.append({
                        "start": word.start,
                        "end": word.end,
                        "text": word.word
                    })
                    word_count += 1

            whisper_segments.append({
                "start": segment.start,
                "end": segment.end,
                "text": segment.text.strip(),
                "words": segment_words
            })

            logger.info("W[%03d] %.2f -> %.2f | %s", index, segment.start, segment.end, segment.text.strip())

        whisper_elapsed = time.perf_counter() - whisper_started

        logger.info("")
        logger.info("Detected language    : %s", info.language)
        logger.info("Language probability : %.4f", info.language_probability)
        logger.info("Audio duration       : %.2f sec", info.duration)
        logger.info("Whisper segments     : %d", len(whisper_segments))
        logger.info("Whisper words        : %d", word_count)
        logger.info("Whisper processing   : %.2f sec", whisper_elapsed)
        logger.info("---------- WHISPER END ------------")

        # ====================================================
        # NEMO SORTFORMER DIARIZATION
        # ====================================================

        logger.info("")
        logger.info("---------- SORTFORMER START -------")

        diarization_started = time.perf_counter()

        if num_speakers is not None:
            logger.info("Requested speakers : %d", num_speakers)
            logger.info("Note: Sortformer auto-detects speakers; num_speakers is kept only for API compatibility")

        # Use the same preprocessed 16 kHz mono WAV as Whisper.
        predicted_segments = diarization_model.diarize(
            audio=[processed_audio_path],
            batch_size=1
        )

        diarization_segments = parse_sortformer_segments(predicted_segments)
        diarization_elapsed = time.perf_counter() - diarization_started

        logger.info("Speaker timeline:")
        for index, item in enumerate(diarization_segments, start=1):
            logger.info(
                "D[%03d] %.2f -> %.2f | %s",
                index,
                item["start"],
                item["end"],
                item["speaker"]
            )

        speakers = sorted({item["speaker"] for item in diarization_segments})

        logger.info("")
        logger.info("Speaker count          : %d", len(speakers))
        logger.info("Speakers               : %s", speakers)
        logger.info("Sortformer processing  : %.2f sec", diarization_elapsed)
        logger.info("---------- SORTFORMER END ---------")

        # ====================================================
        # ASSIGN WHISPER SEGMENT -> SPEAKER
        # ====================================================

        logger.info("")
        logger.info("---------- ASSIGN START -----------")

        assign_started = time.perf_counter()
        result = []
        unknown_count = 0

        for segment in whisper_segments:
            speaker = find_segment_speaker(diarization_segments, segment["start"], segment["end"])

            if speaker == "UNKNOWN":
                unknown_count += 1

            result.append({
                "start": segment["start"],
                "end": segment["end"],
                "speaker": speaker,
                "text": segment["text"]
            })

        assign_elapsed = time.perf_counter() - assign_started

        logger.info("Segments assigned : %d", len(result))
        logger.info("Unknown segments  : %d", unknown_count)
        logger.info("Final segments    : %d", len(result))
        logger.info("Assign time       : %.3f sec", assign_elapsed)
        logger.info("---------- ASSIGN END -------------")

        # ====================================================
        # FINAL TRANSCRIPT
        # ====================================================

        logger.info("")
        logger.info("============== RESULT ==============")

        for index, item in enumerate(result, start=1):
            logger.info(
                "R[%03d] [%.2f - %.2f] %s: %s",
                index,
                item["start"],
                item["end"],
                item["speaker"],
                item["text"]
            )

        logger.info("====================================")

        # ====================================================
        # PERFORMANCE
        # ====================================================

        total_elapsed = time.perf_counter() - request_started
        rtf = total_elapsed / info.duration if info.duration > 0 else 0

        logger.info("")
        logger.info("============= PERFORMANCE =============")
        logger.info("Audio      : %.2f sec", info.duration)
        logger.info("Preprocess : %.2f sec", preprocess_elapsed)
        logger.info("Whisper    : %.2f sec", whisper_elapsed)
        logger.info("Sortformer : %.2f sec", diarization_elapsed)
        logger.info("Assignment : %.3f sec", assign_elapsed)
        logger.info("TOTAL      : %.2f sec", total_elapsed)
        logger.info("RTF        : %.3f", rtf)

        if total_elapsed > 0:
            logger.info("Speed      : %.2fx realtime", info.duration / total_elapsed)

        logger.info("=======================================")

        # ====================================================
        # RESPONSE
        # ====================================================

        return {
            "model": model_name,
            "requestedLanguage": language_value,
            "requestedSpeakerCount": num_speakers,
            "detectedLanguage": info.language,
            "languageProbability": info.language_probability,
            "duration": info.duration,
            "speakerCount": len(speakers),
            "speakers": speakers,
            "wordCount": word_count,
            "processingTime": {
                "preprocess": round(preprocess_elapsed, 3),
                "whisper": round(whisper_elapsed, 3),
                "diarization": round(diarization_elapsed, 3),
                "speakerAssignment": round(assign_elapsed, 3),
                "total": round(total_elapsed, 3),
                "rtf": round(rtf, 4)
            },
            "segments": result
        }

    except HTTPException:
        raise
    except Exception as ex:
        logger.exception("Transcription failed: %s", str(ex))
        raise HTTPException(status_code=500, detail=str(ex))
    finally:
        safe_remove(processed_audio_path)
        safe_remove(audio_path)