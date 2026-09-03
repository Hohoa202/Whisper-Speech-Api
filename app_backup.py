import os
import gc
import time
import logging
import tempfile
import subprocess
import asyncio
from enum import Enum

import torch
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from faster_whisper import WhisperModel
from pyannote.audio import Pipeline
from pyannote.core import Segment


# ============================================================
# LOGGING
# ============================================================

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("speech-api")


# ============================================================
# FASTAPI
# ============================================================

app = FastAPI(title="Whisper Speech API", version="1.1.0")

# Only one GPU job runs at a time. Uploads can still be received concurrently.
gpu_lock = asyncio.Lock()
waiting_jobs = 0
waiting_jobs_lock = asyncio.Lock()

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
# PYANNOTE
# ============================================================

PYANNOTE_MODEL = "./models/pyannote-community-1"
diarization_pipeline = None


def get_diarization_pipeline():
    global diarization_pipeline
    if diarization_pipeline is not None:
        logger.info("Reuse Pyannote model: %s", PYANNOTE_MODEL)
        return diarization_pipeline
    logger.info("Loading Pyannote model...")
    logger.info("Pyannote path: %s", PYANNOTE_MODEL)
    pyannote_start = time.perf_counter()
    diarization_pipeline = Pipeline.from_pretrained(PYANNOTE_MODEL)
    if DEVICE == "cuda":
        diarization_pipeline.to(torch.device("cuda"))
    logger.info("Pyannote loaded in %.2f sec", time.perf_counter() - pyannote_start)
    return diarization_pipeline


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
# SPEAKER MATCHING
# ============================================================

def find_segment_speaker(diarization, start, end):
    if start is None or end is None:
        return "UNKNOWN"

    cropped = diarization.crop(Segment(start, end))
    speakers = cropped.labels()

    if not speakers:
        return "UNKNOWN"

    return max(speakers, key=lambda speaker: cropped.label_duration(speaker))


def remap_speakers_by_first_appearance(items):
    speaker_map = {}
    next_id = 1
    for item in items:
        speaker = item["speaker"]
        if speaker == "UNKNOWN":
            continue
        if speaker not in speaker_map:
            speaker_map[speaker] = f"SPEAKER_{next_id:02d}"
            next_id += 1
        item["speaker"] = speaker_map[speaker]
    return items, speaker_map

def merge_same_speaker_segments(items, split_sentences=True, max_gap=1.0, max_duration=30.0):
    if not items:
        return []

    merged = []
    current = items[0].copy()

    for item in items[1:]:
        gap = item["start"] - current["end"]
        current_text = current["text"].rstrip()
        current_duration = current["end"] - current["start"]
        ends_sentence = current_text.endswith(("。", "！", "？", "!", "?"))

        same_speaker = item["speaker"] == current["speaker"] and item["speaker"] != "UNKNOWN"
        should_split = split_sentences and (gap > max_gap or (current_duration >= max_duration and ends_sentence))
        can_merge = same_speaker and not should_split

        if can_merge:
            current["end"] = item["end"]
            current["text"] = current_text + item["text"].lstrip()
        else:
            merged.append(current)
            current = item.copy()

    merged.append(current)
    return merged


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
    num_speakers: int | None = Form(None),
    speaker_diarization_enabled: bool = Form(True),
    sentence_split_enabled: bool = Form(True),
    max_gap: float = Form(1.0),
    max_duration: float = Form(30.0),
    vad_enabled: bool = Form(True),
    vad_threshold: float = Form(0.2),
    vad_min_speech_ms: int = Form(50),
    vad_min_silence_ms: int = Form(500),
    vad_speech_pad_ms: int = Form(300)
):
    request_started = time.perf_counter()
    audio_path = None
    processed_audio_path = None
    diarization_path = None

    language_value = language.value
    model_name = model.value

    if speaker_diarization_enabled and num_speakers is not None and num_speakers < 1:
        raise HTTPException(status_code=400, detail="num_speakers must be greater than 0")
    if sentence_split_enabled and max_gap < 0:
        raise HTTPException(status_code=400, detail="max_gap must be >= 0")
    if sentence_split_enabled and max_duration <= 0:
        raise HTTPException(status_code=400, detail="max_duration must be greater than 0")
    if not 0 <= vad_threshold <= 1:
        raise HTTPException(status_code=400, detail="vad_threshold must be between 0 and 1")
    if vad_min_speech_ms < 0 or vad_min_silence_ms < 0 or vad_speech_pad_ms < 0:
        raise HTTPException(status_code=400, detail="VAD duration settings must be >= 0")

    logger.info("")
    logger.info("============================================")
    logger.info("NEW TRANSCRIPTION REQUEST")
    logger.info("File     : %s", file.filename)
    logger.info("Model    : %s", model_name)
    logger.info("Language : %s", language_value)
    logger.info("Diarization : %s", "on" if speaker_diarization_enabled else "off")
    logger.info("Speakers    : %s", num_speakers if speaker_diarization_enabled and num_speakers is not None else "auto" if speaker_diarization_enabled else "disabled")
    logger.info("Sentence split : %s", "on" if sentence_split_enabled else "off")
    if sentence_split_enabled:
        logger.info("Merge cfg      : maxGap=%.2fs, maxDuration=%.2fs", max_gap, max_duration)
    logger.info("VAD      : %s", "on" if vad_enabled else "off")
    if vad_enabled:
        logger.info("VAD cfg  : threshold=%.2f, minSpeech=%dms, minSilence=%dms, pad=%dms", vad_threshold, vad_min_speech_ms, vad_min_silence_ms, vad_speech_pad_ms)
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

        global waiting_jobs
        async with waiting_jobs_lock:
            waiting_jobs += 1
            queue_position = waiting_jobs
        logger.info("GPU queue position: %d", queue_position)

        try:
            async with gpu_lock:
                async with waiting_jobs_lock:
                    waiting_jobs -= 1
                queue_wait_elapsed = time.perf_counter() - request_started
                logger.info("GPU acquired after %.2f sec", queue_wait_elapsed)

                # ====================================================
                # AUDIO PREPROCESS
                # ====================================================

                logger.info("")
                logger.info("---------- PREPROCESS START --------")
                preprocess_started = time.perf_counter()
                processed_audio_path = await asyncio.to_thread(preprocess_audio, audio_path)
                preprocess_elapsed = time.perf_counter() - preprocess_started
                logger.info("Processed audio : %s", processed_audio_path)
                logger.info("Filters         : highpass 80Hz + afftdn + dynaudnorm")
                logger.info("Format          : WAV PCM 16-bit / mono / 16 kHz")
                logger.info("Preprocess time : %.2f sec", preprocess_elapsed)
                logger.info("---------- PREPROCESS END ----------")

                # ====================================================
                # LOAD WHISPER
                # ====================================================

                model_instance = await asyncio.to_thread(get_whisper_model, model_name)

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
                transcribe_kwargs = {"language": whisper_language, "beam_size": 5, "word_timestamps": True, "vad_filter": vad_enabled, "condition_on_previous_text": True, "initial_prompt": "句読点（、。！？）を適切に使用して、日本語の文章として文字起こししてください。"}
                # transcribe_kwargs = {"language": whisper_language, "beam_size": 5, "word_timestamps": True, "vad_filter": vad_enabled, "condition_on_previous_text": True}
                if vad_enabled:
                    transcribe_kwargs["vad_parameters"] = {"threshold": vad_threshold, "min_speech_duration_ms": vad_min_speech_ms, "min_silence_duration_ms": vad_min_silence_ms, "speech_pad_ms": vad_speech_pad_ms}

                def run_whisper():
                    segments, info = model_instance.transcribe(processed_audio_path, **transcribe_kwargs)
                    return list(segments), info

                segments, info = await asyncio.to_thread(run_whisper)
                whisper_segments = []
                word_count = 0

                for index, segment in enumerate(segments, start=1):
                    segment_words = []
                    if segment.words:
                        for word in segment.words:
                            if word.start is None or word.end is None:
                                continue
                            segment_words.append({"start": word.start, "end": word.end, "text": word.word})
                            word_count += 1
                    whisper_segments.append({"start": segment.start, "end": segment.end, "text": segment.text.strip(), "words": segment_words})
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
                # CONVERT FOR PYANNOTE
                # ====================================================

                convert_elapsed = 0.0
                diarization_elapsed = 0.0
                diarization = None
                if speaker_diarization_enabled:
                    logger.info("")
                    logger.info("---------- AUDIO CONVERT START -----")
                    convert_started = time.perf_counter()
                    diarization_path = processed_audio_path
                    convert_elapsed = time.perf_counter() - convert_started
                    logger.info("Pyannote audio : %s", diarization_path)
                    logger.info("Format         : WAV PCM 16-bit / mono / 16 kHz")
                    logger.info("Reuse preprocessed WAV (no second conversion)")
                    logger.info("---------- AUDIO CONVERT END -------")

                # ====================================================
                # PYANNOTE
                # ====================================================

                    logger.info("")
                    logger.info("---------- PYANNOTE START ---------")
                    diarization_started = time.perf_counter()
                    pipeline_instance = await asyncio.to_thread(get_diarization_pipeline)

                    def run_diarization():
                        if num_speakers is not None:
                            logger.info("Speaker mode : fixed (%d)", num_speakers)
                            return pipeline_instance(diarization_path, num_speakers=num_speakers)
                        logger.info("Speaker mode : auto")
                        return pipeline_instance(diarization_path)

                    output = await asyncio.to_thread(run_diarization)
                    diarization = output.speaker_diarization
                    diarization_elapsed = time.perf_counter() - diarization_started
                else:
                    logger.info("Pyannote skipped because speaker diarization is disabled")
        except BaseException:
            if gpu_lock.locked() is False:
                async with waiting_jobs_lock:
                    if waiting_jobs > 0:
                        waiting_jobs -= 1
            raise

        # ====================================================
        # LOG SPEAKER TIMELINE
        # ====================================================

        if speaker_diarization_enabled:
            logger.info("Speaker timeline:")
            diarization_count = 0
            for turn, speaker in diarization:
                diarization_count += 1
                logger.info("D[%03d] %.2f -> %.2f | %s", diarization_count, turn.start, turn.end, speaker)
            speakers = diarization.labels()
            logger.info("")
            logger.info("Speaker count : %d", len(speakers))
            logger.info("Speakers      : %s", speakers)
            logger.info("Pyannote processing: %.2f sec", diarization_elapsed)
            logger.info("---------- PYANNOTE END -----------")
        else:
            speakers = []

        # ====================================================
        # ASSIGN WHISPER SEGMENT -> SPEAKER
        # ====================================================

        assign_elapsed = 0.0
        speaker_map = {}
        if speaker_diarization_enabled:
            logger.info("")
            logger.info("---------- ASSIGN START -----------")
            assign_started = time.perf_counter()
            result = []
            unknown_count = 0
            for segment in whisper_segments:
                speaker = find_segment_speaker(diarization, segment["start"], segment["end"])
                if speaker == "UNKNOWN":
                    unknown_count += 1
                result.append({"start": segment["start"], "end": segment["end"], "speaker": speaker, "text": segment["text"]})
            raw_result_count = len(result)
            result, speaker_map = remap_speakers_by_first_appearance(result)
            result = merge_same_speaker_segments(result, split_sentences=sentence_split_enabled, max_gap=max_gap, max_duration=max_duration)
            assign_elapsed = time.perf_counter() - assign_started
            logger.info("Segments assigned : %d", raw_result_count)
            logger.info("Speaker remap     : %s", speaker_map)
            logger.info("Unknown segments  : %d", unknown_count)
            logger.info("Final segments    : %d", len(result))
            logger.info("Assign time       : %.3f sec", assign_elapsed)
            logger.info("---------- ASSIGN END -------------")
        else:
            result = whisper_segments
            logger.info("Speaker assignment and merge skipped; returning original Whisper segments")

        # ====================================================
        # FINAL TRANSCRIPT
        # ====================================================

        logger.info("")
        logger.info("============== RESULT ==============")

        for index, item in enumerate(result, start=1):
            if speaker_diarization_enabled:
                logger.info("R[%03d] [%.2f - %.2f] %s: %s", index, item["start"], item["end"], item["speaker"], item["text"])
            else:
                logger.info("R[%03d] [%.2f - %.2f] %s", index, item["start"], item["end"], item["text"])

        logger.info("====================================")

        # ====================================================
        # PERFORMANCE
        # ====================================================

        total_elapsed = time.perf_counter() - request_started
        rtf = total_elapsed / info.duration if info.duration > 0 else 0

        logger.info("")
        logger.info("============= PERFORMANCE =============")
        logger.info("Queue wait : %.2f sec", queue_wait_elapsed)
        logger.info("Audio      : %.2f sec", info.duration)
        logger.info("Preprocess : %.2f sec", preprocess_elapsed)
        logger.info("Whisper    : %.2f sec", whisper_elapsed)
        logger.info("Conversion : %.2f sec", convert_elapsed)
        logger.info("Pyannote   : %.2f sec", diarization_elapsed)
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
            "speakerDiarizationEnabled": speaker_diarization_enabled,
            "sentenceSplit": {
                "enabled": sentence_split_enabled,
                "maxGap": max_gap if sentence_split_enabled else None,
                "maxDuration": max_duration if sentence_split_enabled else None
            },
            "vad": {
                "enabled": vad_enabled,
                "threshold": vad_threshold if vad_enabled else None,
                "minSpeechDurationMs": vad_min_speech_ms if vad_enabled else None,
                "minSilenceDurationMs": vad_min_silence_ms if vad_enabled else None,
                "speechPadMs": vad_speech_pad_ms if vad_enabled else None
            },
            "detectedLanguage": info.language,
            "languageProbability": info.language_probability,
            "duration": info.duration,
            "speakerCount": len(speaker_map),
            "speakers": list(speaker_map.values()),
            "wordCount": word_count,
            "processingTime": {
                "queueWait": round(queue_wait_elapsed, 3),
                "preprocess": round(preprocess_elapsed, 3),
                "whisper": round(whisper_elapsed, 3),
                "audioConversion": round(convert_elapsed, 3),
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
        safe_remove(diarization_path)
        safe_remove(processed_audio_path)
        safe_remove(audio_path)
