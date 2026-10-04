"""
Image Preprocessor for Vision Inference.

Resizes oversized images to bounded dimensions before sending to any vision
provider (Gemini, Ollama, etc.).
Preserves the original image file and path untouched.
"""
import os
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Generator
from PIL import Image

from app.config.logging import get_logger

logger = get_logger(__name__)


@contextmanager
def preprocess_for_vision(
    image_path: str | Path,
    max_dimension: int = 1920,
) -> Generator[str, None, None]:
    """
    Context manager that yields an image path suitable for vision LLM inference.

    If the image exceeds max_dimension along either width or height, it is resized
    preserving aspect ratio and saved to a temporary file.
    The temporary file is cleaned up when exiting the context.

    CRITICAL: The original image at image_path is NEVER modified.
    """
    orig_path = Path(image_path)
    if not orig_path.exists():
        yield str(image_path)
        return

    temp_path: str | None = None
    try:
        with Image.open(orig_path) as img:
            w, h = img.size
            max_side = max(w, h)

            if max_side > max_dimension:
                scale = max_dimension / float(max_side)
                new_w = max(1, int(w * scale))
                new_h = max(1, int(h * scale))

                logger.info(
                    "vision.image_resize",
                    original_size=(w, h),
                    target_size=(new_w, new_h),
                    max_dimension=max_dimension,
                    image=str(orig_path),
                )

                if img.mode not in ("RGB", "RGBA", "L"):
                    resized_img = img.convert("RGB").resize((new_w, new_h), Image.Resampling.LANCZOS)
                else:
                    resized_img = img.resize((new_w, new_h), Image.Resampling.LANCZOS)

                temp_fd, temp_path = tempfile.mkstemp(prefix="vision_input_", suffix=".png")
                os.close(temp_fd)
                resized_img.save(temp_path, format="PNG", optimize=True)

    except Exception as exc:
        logger.warning(
            "vision.preprocess_fallback",
            image=str(orig_path),
            error=str(exc),
        )
        temp_path = None

    target_path = temp_path if temp_path else str(orig_path)
    try:
        yield target_path
    finally:
        if temp_path and os.path.exists(temp_path):
            try:
                os.unlink(temp_path)
            except OSError:
                pass
