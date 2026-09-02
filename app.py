import os
import gc
import time
import logging
import tempfile
from enum import Enum

import torch
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from faster_whisper import WhisperModel
from pyannote.audio import Pipeline
from pyannote.core import Segment


# ============================================================
# LOGGING
# ============================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

logger = logging.getLogger("speech-api")


# ============================================================
# FASTAPI
# ============================================================

app = FastAPI(
    title="Whisper Speech API",
    version="1.0.0"
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

COMPUTE_TYPE = (
    "float16"
    if DEVICE == "cuda"
    else "int8"
)

logger.info("============================================")
logger.info("Speech API starting")
logger.info("Device       : %s", DEVICE)
logger.info("Compute type : %s", COMPUTE_TYPE)

if DEVICE == "cuda":
    logger.info(
        "GPU          : %s",
        torch.cuda.get_device_name(0)
    )

    logger.info(
        "VRAM         : %.2f GB",
        torch.cuda.get_device_properties(0).total_memory / 1024**3
    )

logger.info("============================================")


# ============================================================
# PYANNOTE
# ============================================================

PYANNOTE_MODEL = "./models/pyannote-community-1"

logger.info("Loading Pyannote model...")
logger.info("Pyannote path: %s", PYANNOTE_MODEL)

pyannote_start = time.perf_counter()

diarization_pipeline = Pipeline.from_pretrained(
    PYANNOTE_MODEL
)

if DEVICE == "cuda":
    diarization_pipeline.to(
        torch.device("cuda")
    )

logger.info(
    "Pyannote loaded in %.2f sec",
    time.perf_counter() - pyannote_start
)


# ============================================================
# WHISPER MODEL
# ============================================================

whisper_model = None
current_whisper_model = None


def get_whisper_model(model_name: str):

    global whisper_model
    global current_whisper_model

    # Reuse model hiện tại
    if (
        whisper_model is not None
        and current_whisper_model == model_name
    ):
        logger.info(
            "Reuse Whisper model: %s",
            model_name
        )

        return whisper_model

    # Đổi model -> unload model cũ
    if whisper_model is not None:

        logger.info(
            "Unload Whisper model: %s",
            current_whisper_model
        )

        del whisper_model

        whisper_model = None
        current_whisper_model = None

        gc.collect()

        if DEVICE == "cuda":
            torch.cuda.empty_cache()

    logger.info(
        "Loading Whisper model: %s",
        model_name
    )

    started = time.perf_counter()

    whisper_model = WhisperModel(
        model_name,
        device=DEVICE,
        compute_type=COMPUTE_TYPE
    )

    current_whisper_model = model_name

    logger.info(
        "Whisper model %s loaded in %.2f sec",
        model_name,
        time.perf_counter() - started
    )

    return whisper_model


# ============================================================
# SPEAKER MATCHING
# ============================================================

def find_speaker(
    diarization,
    start,
    end
):

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
        key=lambda speaker:
            cropped.label_duration(speaker)
    )


# ============================================================
# MERGE CONSECUTIVE WORDS OF SAME SPEAKER
# ============================================================

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

            current["text"] = (
                current["text"].strip()
            )

            result.append(current)

            current = {
                "start": word["start"],
                "end": word["end"],
                "speaker": word["speaker"],
                "text": word["text"]
            }

    current["text"] = (
        current["text"].strip()
    )

    result.append(current)

    return result


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
        "models": [
            "small",
            "large-v3",
            "large-v3-turbo"
        ],
        "languages": [
            "auto",
            "ja",
            "en",
            "vi"
        ]
    }


# ============================================================
# TRANSCRIBE
# ============================================================

@app.post("/transcribe")
async def transcribe(
    file: UploadFile = File(...),

    language: LanguageName = Form(
        LanguageName.ja
    ),

    model: WhisperModelName = Form(
        WhisperModelName.large_v3_turbo
    )
):

    request_started = time.perf_counter()

    audio_path = None

    language_value = language.value
    model_name = model.value

    logger.info("")
    logger.info("============================================")
    logger.info("NEW TRANSCRIPTION REQUEST")
    logger.info("File     : %s", file.filename)
    logger.info("Model    : %s", model_name)
    logger.info("Language : %s", language_value)
    logger.info("Device   : %s", DEVICE)
    logger.info("============================================")

    try:

        # ====================================================
        # SAVE FILE
        # ====================================================

        suffix = os.path.splitext(
            file.filename
        )[1]

        data = await file.read()

        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=suffix
        ) as temp:

            temp.write(data)

            audio_path = temp.name

        logger.info(
            "File size : %.2f MB",
            len(data) / 1024 / 1024
        )

        logger.info(
            "Temp file : %s",
            audio_path
        )


        # ====================================================
        # LOAD WHISPER
        # ====================================================

        model_instance = get_whisper_model(
            model_name
        )


        # ====================================================
        # LANGUAGE
        # ====================================================

        whisper_language = (
            None
            if language_value == "auto"
            else language_value
        )


        # ====================================================
        # WHISPER
        # ====================================================

        logger.info("")
        logger.info(
            "---------- WHISPER START ----------"
        )

        whisper_started = time.perf_counter()

        segments, info = model_instance.transcribe(
            audio_path,
            language=whisper_language,
            beam_size=5,
            word_timestamps=True,
            vad_filter=True
        )

        whisper_segments = []

        word_count = 0

        #
        # faster-whisper trả generator.
        # Inference thực sự xảy ra khi loop ở đây.
        #
        for index, segment in enumerate(
            segments,
            start=1
        ):

            segment_words = []

            if segment.words:

                for word in segment.words:

                    if (
                        word.start is None
                        or word.end is None
                    ):
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

            logger.info(
                "W[%03d] %.2f -> %.2f | %s",
                index,
                segment.start,
                segment.end,
                segment.text.strip()
            )

        whisper_elapsed = (
            time.perf_counter()
            - whisper_started
        )

        logger.info("")
        logger.info(
            "Detected language    : %s",
            info.language
        )

        logger.info(
            "Language probability : %.4f",
            info.language_probability
        )

        logger.info(
            "Audio duration       : %.2f sec",
            info.duration
        )

        logger.info(
            "Whisper segments     : %d",
            len(whisper_segments)
        )

        logger.info(
            "Whisper words        : %d",
            word_count
        )

        logger.info(
            "Whisper processing   : %.2f sec",
            whisper_elapsed
        )

        logger.info(
            "---------- WHISPER END ------------"
        )


        # ====================================================
        # PYANNOTE
        # ====================================================

        logger.info("")
        logger.info(
            "---------- PYANNOTE START ---------"
        )

        diarization_started = (
            time.perf_counter()
        )

        output = diarization_pipeline(
            audio_path
        )

        diarization = (
            output.speaker_diarization
        )

        diarization_elapsed = (
            time.perf_counter()
            - diarization_started
        )


        # ====================================================
        # LOG SPEAKER TIMELINE
        # ====================================================

        logger.info("Speaker timeline:")

        diarization_count = 0

        for turn, speaker in diarization:

            diarization_count += 1

            logger.info(
                "D[%03d] %.2f -> %.2f | %s",
                diarization_count,
                turn.start,
                turn.end,
                speaker
            )

        speakers = diarization.labels()

        logger.info("")
        logger.info(
            "Speaker count : %d",
            len(speakers)
        )

        logger.info(
            "Speakers      : %s",
            speakers
        )

        logger.info(
            "Pyannote processing: %.2f sec",
            diarization_elapsed
        )

        logger.info(
            "---------- PYANNOTE END -----------"
        )


        # ====================================================
        # ASSIGN WORD -> SPEAKER
        # ====================================================

        logger.info("")
        logger.info(
            "---------- ASSIGN START -----------"
        )

        assign_started = (
            time.perf_counter()
        )

        words = []

        unknown_count = 0

        for segment in whisper_segments:

            for word in segment["words"]:

                speaker = find_speaker(
                    diarization,
                    word["start"],
                    word["end"]
                )

                if speaker == "UNKNOWN":
                    unknown_count += 1

                words.append({
                    "start": word["start"],
                    "end": word["end"],
                    "speaker": speaker,
                    "text": word["text"]
                })

        result = merge_words(words)

        assign_elapsed = (
            time.perf_counter()
            - assign_started
        )

        logger.info(
            "Words assigned : %d",
            len(words)
        )

        logger.info(
            "Unknown words  : %d",
            unknown_count
        )

        logger.info(
            "Final segments : %d",
            len(result)
        )

        logger.info(
            "Assign time    : %.3f sec",
            assign_elapsed
        )

        logger.info(
            "---------- ASSIGN END -------------"
        )


        # ====================================================
        # FINAL TRANSCRIPT
        # ====================================================

        logger.info("")
        logger.info(
            "============== RESULT =============="
        )

        for index, item in enumerate(
            result,
            start=1
        ):

            logger.info(
                "R[%03d] [%.2f - %.2f] %s: %s",
                index,
                item["start"],
                item["end"],
                item["speaker"],
                item["text"]
            )

        logger.info(
            "===================================="
        )


        # ====================================================
        # PERFORMANCE
        # ====================================================

        total_elapsed = (
            time.perf_counter()
            - request_started
        )

        rtf = (
            total_elapsed / info.duration
            if info.duration > 0
            else 0
        )

        logger.info("")
        logger.info(
            "============= PERFORMANCE ============="
        )

        logger.info(
            "Audio      : %.2f sec",
            info.duration
        )

        logger.info(
            "Whisper    : %.2f sec",
            whisper_elapsed
        )

        logger.info(
            "Pyannote   : %.2f sec",
            diarization_elapsed
        )

        logger.info(
            "Assignment : %.3f sec",
            assign_elapsed
        )

        logger.info(
            "TOTAL      : %.2f sec",
            total_elapsed
        )

        logger.info(
            "RTF        : %.3f",
            rtf
        )

        if total_elapsed > 0:

            logger.info(
                "Speed      : %.2fx realtime",
                info.duration / total_elapsed
            )

        logger.info(
            "======================================="
        )


        # ====================================================
        # RESPONSE
        # ====================================================

        return {
            "model": model_name,

            "requestedLanguage":
                language_value,

            "detectedLanguage":
                info.language,

            "languageProbability":
                info.language_probability,

            "duration":
                info.duration,

            "speakerCount":
                len(speakers),

            "speakers":
                speakers,

            "wordCount":
                word_count,

            "processingTime": {
                "whisper":
                    round(whisper_elapsed, 3),

                "diarization":
                    round(diarization_elapsed, 3),

                "speakerAssignment":
                    round(assign_elapsed, 3),

                "total":
                    round(total_elapsed, 3),

                "rtf":
                    round(rtf, 4)
            },

            "segments":
                result
        }

    except Exception as ex:

        logger.exception(
            "Transcription failed: %s",
            str(ex)
        )

        raise HTTPException(
            status_code=500,
            detail=str(ex)
        )

    finally:

        if (
            audio_path
            and os.path.exists(audio_path)
        ):

            os.remove(audio_path)

            logger.info(
                "Deleted temp file: %s",
                audio_path
            )