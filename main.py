import os
import io
import base64
import traceback
from typing import Optional, List, Tuple

import httpx
from PIL import Image
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="Saree AI API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------- CONFIG ----------------
MAX_FILE_SIZE = 8 * 1024 * 1024   # 8 MB per upload before compression
HTTP_TIMEOUT = 120.0              # gpt-image-1 can be slower than text models
OUTPUT_SIZE = "1024x1536"         # portrait is usually better for fashion output
OUTPUT_QUALITY = "high"           # low | medium | high | auto
MODEL_NAME = "gpt-image-1"

# ---------------- HELPERS ----------------
def compress_to_jpeg(image_bytes: bytes, max_dim: int = 1600, quality: int = 82) -> bytes:
    """
    Normalize uploaded images to JPEG and resize them to keep payloads reasonable.
    Falls back to original bytes if PIL cannot process the image.
    """
    try:
        with Image.open(io.BytesIO(image_bytes)) as img:
            # Convert to RGB for JPEG compatibility
            if img.mode not in ("RGB", "L"):
                img = img.convert("RGB")
            elif img.mode == "L":
                img = img.convert("RGB")

            width, height = img.size
            largest = max(width, height)

            if largest > max_dim:
                ratio = max_dim / float(largest)
                new_size = (max(1, int(width * ratio)), max(1, int(height * ratio)))
                img = img.resize(new_size, Image.LANCZOS)

            buffer = io.BytesIO()
            img.save(buffer, format="JPEG", quality=quality, optimize=True)
            return buffer.getvalue()
    except Exception:
        return image_bytes


async def read_and_validate_file(file: UploadFile, field_name: str) -> bytes:
    """
    Read uploaded file, validate presence and size, then compress/normalize it.
    """
    if not file or not file.filename:
        raise ValueError(f"{field_name} is missing")

    content = await file.read()

    if not content:
        raise ValueError(f"{field_name} is empty")

    if len(content) > MAX_FILE_SIZE:
        raise ValueError(f"{field_name} is too large. Max allowed size is {MAX_FILE_SIZE // (1024 * 1024)} MB")

    return compress_to_jpeg(content)


def build_edit_prompt(drape_style: str, has_user_photo: bool) -> str:
    """
    Prompt for GPT Image edit using the uploaded reference images directly.
    Keep it focused and explicit.
    """
    person_block = (
        "Use the uploaded user photo as the person reference. Preserve the same face, skin tone, body proportions, and overall likeness as closely as possible. "
        "Dress that same person in the saree using the uploaded saree references."
        if has_user_photo
        else
        "Create a realistic elegant Indian female model wearing the saree from the uploaded references."
    )

    return (
        f"Create a highly realistic full-body fashion image. "
        f"Use the uploaded images as the source references for the saree, blouse, fabric details, colors, border work, motifs, and overall styling. "
        f"{person_block} "
        f"Drape the saree in {drape_style} style with realistic pleats, natural fabric fall, and accurate border placement. "
        f"The blouse should match the uploaded blouse reference as closely as possible. "
        f"Keep the saree design faithful to the uploaded references, with strong preservation of visible patterns and decorative work. "
        f"Studio fashion photography, neutral light background, sharp focus, premium catalog look, realistic anatomy, realistic fabric physics."
    )


def make_image_part(field_name: str, filename: str, data: bytes, mime_type: str = "image/jpeg") -> Tuple[str, Tuple[str, bytes, str]]:
    """
    Format a multipart file tuple for httpx.
    """
    return (field_name, (filename, data, mime_type))


# ---------------- HEALTH ----------------
@app.get("/")
def root():
    api_key = os.environ.get("OPENAI_API_KEY", "")
    return {
        "status": "Saree AI API is running",
        "openai_key_set": bool(api_key),
        "key_length": len(api_key),
        "model": MODEL_NAME
    }


@app.get("/health")
def health():
    return {"status": "ok"}


# ---------------- MAIN ENDPOINT ----------------
@app.post("/generate-saree")
async def generate_saree(
    pallu_image: UploadFile = File(...),
    blouse_image: UploadFile = File(...),
    fabric_image: UploadFile = File(...),
    user_photo: Optional[UploadFile] = File(None),
    drape_style: str = Form("Bengali"),
):
    try:
        api_key = os.environ.get("OPENAI_API_KEY", "").strip()
        if not api_key:
            return JSONResponse(
                status_code=500,
                content={"error": "OPENAI_API_KEY is not set in environment variables."}
            )

        # Read and validate uploaded files
        pallu_bytes = await read_and_validate_file(pallu_image, "pallu_image")
        blouse_bytes = await read_and_validate_file(blouse_image, "blouse_image")
        fabric_bytes = await read_and_validate_file(fabric_image, "fabric_image")

        has_user_photo = False
        user_bytes: Optional[bytes] = None

        if user_photo and user_photo.filename:
            user_bytes = await read_and_validate_file(user_photo, "user_photo")
            has_user_photo = True

        prompt = build_edit_prompt(drape_style=drape_style, has_user_photo=has_user_photo)

        # Build multipart form data for /v1/images/edits
        # GPT Image edit accepts one or more source images.
        files: List[Tuple[str, Tuple[Optional[str], object, Optional[str]]]] = [
            ("model", (None, MODEL_NAME, None)),
            ("prompt", (None, prompt, None)),
            ("size", (None, OUTPUT_SIZE, None)),
            ("quality", (None, OUTPUT_QUALITY, None)),
            ("input_fidelity", (None, "high", None)),
            ("n", (None, "1", None)),
            ("output_format", (None, "png", None)),
        ]

        # OpenAI docs specify the parameter name as `image` for one or more source images.
        files.append(make_image_part("image", "pallu.jpg", pallu_bytes))
        files.append(make_image_part("image", "blouse.jpg", blouse_bytes))
        files.append(make_image_part("image", "fabric.jpg", fabric_bytes))

        if has_user_photo and user_bytes:
            files.append(make_image_part("image", "user_photo.jpg", user_bytes))

        print("Starting OpenAI image edit request...")
        print(f"Using model={MODEL_NAME}, has_user_photo={has_user_photo}, drape_style={drape_style}")

        async with httpx.AsyncClient(timeout=HTTP_TIMEOUT) as client:
            response = await client.post(
                "https://api.openai.com/v1/images/edits",
                headers={
                    "Authorization": f"Bearer {api_key}"
                },
                files=files,
            )

        # Safely handle non-JSON responses too
        raw_text = response.text

        if response.status_code != 200:
            return JSONResponse(
                status_code=500,
                content={
                    "error": "OpenAI image edit request failed",
                    "status_code": response.status_code,
                    "details": raw_text
                }
            )

        try:
            result = response.json()
        except Exception:
            return JSONResponse(
                status_code=500,
                content={
                    "error": "OpenAI returned a non-JSON response",
                    "details": raw_text
                }
            )

        data = result.get("data") or []
        if not data:
            return JSONResponse(
                status_code=500,
                content={
                    "error": "No image data returned from OpenAI",
                    "response": result
                }
            )

        first = data[0]
        image_b64 = first.get("b64_json")

        if not image_b64:
            return JSONResponse(
                status_code=500,
                content={
                    "error": "OpenAI response did not include b64_json",
                    "response": result
                }
            )

        return JSONResponse(content={
            "success": True,
            "model_used": MODEL_NAME,
            "used_uploaded_images": True,
            "used_user_photo": has_user_photo,
            "prompt_used": prompt,
            "mime_type": "image/png",
            "image_base64": image_b64
        })

    except ValueError as ve:
        return JSONResponse(
            status_code=400,
            content={"error": str(ve)}
        )

    except Exception as e:
        detail = traceback.format_exc()
        print("ERROR:\n", detail)
        return JSONResponse(
            status_code=500,
            content={
                "error": str(e),
                "traceback": detail
            }
        )