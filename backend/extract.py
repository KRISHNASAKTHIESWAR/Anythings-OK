"""
backend/extract.py
Multimodal file extraction — image, audio, text.
Returns raw text strings (not LlamaIndex Documents).
"""

import os
import logging
import json
import requests
from typing import List, Dict

logger = logging.getLogger(__name__)


def _ollama_chat(model: str, messages: list, timeout: int = 120) -> str:
    """Direct Ollama API call — no LlamaIndex dependency."""
    resp = requests.post(
        "http://localhost:11434/api/chat",
        json={"model": model, "messages": messages, "stream": False},
        timeout=timeout,
    )
    resp.raise_for_status()
    return resp.json()["message"]["content"]


def _ollama_vision(model: str, prompt: str, image_path: str, timeout: int = 180) -> str:
    """Ollama vision call with base64 image."""
    import base64

    with open(image_path, "rb") as f:
        img_b64 = base64.b64encode(f.read()).decode("utf-8")

    resp = requests.post(
        "http://localhost:11434/api/chat",
        json={
            "model": model,
            "messages": [
                {"role": "user", "content": prompt, "images": [img_b64]}
            ],
            "stream": False,
        },
        timeout=timeout,
    )
    resp.raise_for_status()
    return resp.json()["message"]["content"]


# ── Extractors ──────────────────────────────────────────────────────

def extract_image(path: str) -> List[Dict]:
    logger.info(f"[EXTRACT] Image: {path}")
    prompt = (
        "Describe this image in detailed text so someone can fully understand it.\n"
        "If it contains readable text, reproduce it exactly.\n"
        "If it has diagrams, tables, or charts, describe their structure."
    )
    text = _ollama_vision("qwen3-vl:4b", prompt, path)
    return [{"text": text.strip(), "source": path, "type": "image"}]


def extract_audio(path: str) -> List[Dict]:
    """
    Transcribe audio using Moonshine (CPU, no CUDA needed).
    Supports .wav natively. For .mp3/.m4a/.flac, converts to wav first via ffmpeg.
    """
    logger.info(f"[EXTRACT] Audio: {path}")
    import subprocess, tempfile

    ext = os.path.splitext(path)[1].lower()
    wav_path = path

    # Moonshine expects wav — convert other formats via ffmpeg
    if ext != ".wav":
        logger.info(f"[EXTRACT] Converting {ext} to wav via ffmpeg...")
        tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        tmp.close()
        wav_path = tmp.name
        try:
            subprocess.run(
                ["ffmpeg", "-y", "-i", path, "-ar", "16000", "-ac", "1", wav_path],
                capture_output=True, check=True,
            )
        except FileNotFoundError:
            raise RuntimeError(
                "ffmpeg not found. Install it: https://ffmpeg.org/download.html"
            )

    try:
        from moonshine_voice import (
            Transcriber, TranscriptEventListener,
            get_model_for_language, load_wav_file,
        )

        model_path, model_arch = get_model_for_language("en")
        transcriber = Transcriber(model_path=model_path, model_arch=model_arch)
        stream = transcriber.create_stream(update_interval=0.5)

        # Collect completed lines
        lines = []

        class FileListener(TranscriptEventListener):
            def on_line_completed(self, event):
                lines.append(event.line.text)

        stream.add_listener(FileListener())
        stream.start()

        # Feed audio in chunks
        audio_data, sample_rate = load_wav_file(wav_path)
        chunk_size = int(0.5 * sample_rate)  # 500ms chunks
        for i in range(0, len(audio_data), chunk_size):
            chunk = audio_data[i : i + chunk_size]
            stream.add_audio(chunk, sample_rate)

        stream.stop()
        transcriber.stop()
        transcript = " ".join(lines).strip()

    except ImportError:
        # Fallback: try the simpler useful-moonshine-onnx package
        logger.info("[EXTRACT] moonshine-voice not found, trying useful-moonshine-onnx...")
        try:
            from moonshine_onnx import MoonshineOnnxModel, load_audio
            model = MoonshineOnnxModel(model_name="moonshine/base")
            audio = load_audio(wav_path)
            tokens = model.generate(audio)
            transcript = model.tokenizer.decode_batch(tokens)[0]
        except ImportError:
            raise ImportError(
                "No audio transcription library found. Install one:\n"
                "  pip install moonshine-voice\n"
                "  pip install useful-moonshine-onnx"
            )
    finally:
        # Clean up temp wav if we created one
        if wav_path != path and os.path.exists(wav_path):
            os.unlink(wav_path)

    logger.info(f"[EXTRACT] Transcribed {len(transcript)} chars")
    return [{"text": transcript, "source": path, "type": "audio"}]


def extract_text(path: str) -> List[Dict]:
    """
    Extract text from documents using LlamaIndex's SimpleDirectoryReader.
    Supports: .pdf, .txt, .md, .csv, .docx, .pptx, .epub, .html, .xlsx, and more.
    """
    logger.info(f"[EXTRACT] Text: {path}")

    from llama_index.core import SimpleDirectoryReader

    docs = SimpleDirectoryReader(input_files=[path]).load_data()

    results = []
    for doc in docs:
        text = doc.text.strip()
        if text:
            results.append({
                "text": text,
                "source": path,
                "type": "document",
            })

    if not results:
        raise ValueError(f"No text extracted from {path}")

    logger.info(f"[EXTRACT] LlamaIndex extracted {len(results)} document(s)")
    return results


def extract(file_path: str) -> List[Dict]:
    """Route to correct extractor based on extension."""
    ext = os.path.splitext(file_path)[1].lower()

    if ext in (".png", ".jpg", ".jpeg", ".webp"):
        return extract_image(file_path)
    elif ext in (".wav", ".mp3", ".m4a", ".flac"):
        return extract_audio(file_path)
    else:
        # Everything else goes through LlamaIndex's SimpleDirectoryReader
        # which handles: .pdf, .txt, .md, .csv, .docx, .pptx, .epub,
        # .html, .xlsx, .json, .ipynb, and more
        return extract_text(file_path)