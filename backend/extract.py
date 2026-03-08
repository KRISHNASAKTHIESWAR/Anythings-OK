import os
import logging
from typing import List

from llama_index.core.llms import ChatMessage, TextBlock, ImageBlock
from llama_index.llms.ollama import Ollama
from llama_index.core.schema import Document
from llama_index.core import SimpleDirectoryReader

from faster_whisper import WhisperModel

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG, format='%(name)s - %(levelname)s - %(message)s')

def extract_image(image_path: str) -> List[Document]:
    """
    Extract detailed textual description from an image using a vision model.
    """
    logger.info(f"[EXTRACT] Starting image extraction: {image_path}")

    try:
        llm = Ollama(
            model="qwen3-vl:4b",
            request_timeout=120.0,
            context_window=8000,
        )

        messages = [
            ChatMessage(
                role="user",
                blocks=[
                    TextBlock(
                        text=(
                            "Describe this image in very detailed text so that someone "
                            "can fully understand the image without seeing it.\n\n"
                            "If the image contains any readable text, reproduce the text "
                            "exactly as written.\n\n"
                            "If the image contains diagrams, tables, charts, UI elements, "
                            "or structured layouts, describe their structure clearly in text."
                        )
                    ),
                    ImageBlock(path=image_path),
                ],
            ),
        ]

        resp = llm.chat(messages)
        logger.debug(f"[EXTRACT] Image LLM response: {len(resp.text)} chars")

        doc = Document(
            text=resp.text.strip(),
            metadata={
                "source": image_path,
                "type": "image",
            },
        )

        logger.info(f"[EXTRACT] Image extraction successful: 1 document")
        return [doc]
    except Exception as e:
        logger.error(f"[EXTRACT] Image extraction failed: {str(e)}", exc_info=True)
        raise


def extract_audio(audio_path: str) -> List[Document]:
    """
    Transcribe audio using Faster-Whisper
    """
    logger.info(f"[EXTRACT] Starting audio extraction: {audio_path}")

    try:
        # CPU version 
        # model = WhisperModel(
        #     "base",
        #     device="cpu",
        #     compute_type="int8"
        # )

        # GPU version 
        model = WhisperModel(
            "base",
            device="cuda",
            compute_type="float16"
        )

        segments, _ = model.transcribe(audio_path)
        logger.debug(f"[EXTRACT] Audio transcribed with {len(list(segments))} segments")

        transcript = ""

        for seg in segments:
            transcript += seg.text + " "

        logger.debug(f"[EXTRACT] Total transcript length: {len(transcript)} chars")

        doc = Document(
            text=transcript.strip(),
            metadata={
                "source": audio_path,
                "type": "audio",
            },
        )

        logger.info(f"[EXTRACT] Audio extraction successful: 1 document")
        return [doc]
    except Exception as e:
        logger.error(f"[EXTRACT] Audio extraction failed: {str(e)}", exc_info=True)
        raise


def extract_text(file_path: str) -> List[Document]:
    """
    Extract text from standard document formats using LlamaIndex readers
    """
    logger.info(f"[EXTRACT] Starting text extraction: {file_path}")

    try:
        docs = SimpleDirectoryReader(
            input_files=[file_path]
        ).load_data()

        logger.debug(f"[EXTRACT] Loaded {len(docs)} documents from file")

        for d in docs:
            d.metadata["source"] = file_path
            d.metadata["type"] = "document"

        logger.info(f"[EXTRACT] Text extraction successful: {len(docs)} documents")
        return docs
    except Exception as e:
        logger.error(f"[EXTRACT] Text extraction failed: {str(e)}", exc_info=True)
        raise


def extract(file_path: str) -> List[Document]:
    """
    Detect file type and route to correct extractor
    """
    logger.info(f"[EXTRACT] Starting extraction for: {file_path}")

    ext = os.path.splitext(file_path)[1].lower()
    logger.debug(f"[EXTRACT] Detected file extension: {ext}")

    try:
        if ext in [".png", ".jpg", ".jpeg"]:
            result = extract_image(file_path)
        elif ext in [".wav", ".mp3", ".m4a", ".flac"]:
            result = extract_audio(file_path)
        elif ext in [".pdf", ".txt", ".docx", ".csv", ".xlsx", ".md"]:
            result = extract_text(file_path)
        else:
            raise ValueError(f"Unsupported file type: {ext}")
        
        logger.info(f"[EXTRACT] Extraction complete: {len(result)} documents returned")
        return result
    except Exception as e:
        logger.error(f"[EXTRACT] Extraction failed: {str(e)}", exc_info=True)
        raise