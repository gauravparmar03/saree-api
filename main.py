import os
import io
import base64
import traceback
from typing import Optional

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

# -------- CONFIG --------
MAX_FILE_SIZE = 5 * 1024 * 1024  # 5MB
TIMEOUT = 180.0  # gpt-image-1 is slow


# -------- IMAGE COMPRESS --------
def compress_image(image_bytes: bytes) -> bytes:
    try:
        img = Image.open(io.BytesIO(image_bytes))
        img = img.convert("RGB")

        # Resize (important)
        img.thumbnail((1024, 1024))

        buffer = io.BytesIO()
        img.save(buffer, format="JPEG", quality=75)
        return buffer.getvalue()
    except Exception:
        return image_bytes


# -------- FILE VALIDATION --------
async def read_file(file: UploadFile, name: str):
    if not file or not file.filename:
        raise Exception(f"{name} missing")

    content = await file.read()

    if not content:
        raise Exception(f"{name} empty")

    if len(content) > MAX_FILE_SIZE:
        raise Exception(f"{name} too large (max 5MB)")

    return compress_image(content)


# -------- HEALTH --------
@app.get("/")
def root():
    return {"status": "API running"}


@app.get("/health")
def health():
    return {"status": "ok"}


# -------- MAIN API --------
@app.post("/generate-saree")
async def generate_saree(
    pallu_image: UploadFile = File(...),
    blouse_image: UploadFile = File(...),
    fabric_image: UploadFile = File(...),
    user_photo: Optional[UploadFile] = File(None),
    drape_style: str = Form("Bengali"),
):
    try:
        api_key = os.environ.get("OPENAI_API_KEY", "")
        if not api_key:
            return JSONResponse(
                status_code=500,
                content={"error": "OPENAI_API_KEY not set"}
            )

        # -------- READ FILES --------
        pallu_bytes = await read_file(pallu_image, "pallu_image")
        blouse_bytes = await read_file(blouse_image, "blouse_image")
        fabric_bytes = await read_file(fabric_image, "fabric_image")

        has_user_photo = False
        user_bytes = None

        if user_photo and user_photo.filename:
            user_bytes = await read_file(user_photo, "user_photo")
            has_user_photo = True

        print("Files received successfully")

        # -------- PROMPT --------
        prompt = f"""
Use the uploaded images to generate a realistic saree model.

- Keep exact fabric, color, and design from images
- Use {drape_style} drape style
- Realistic pleats and pallu
- High quality fashion photography

{"Use the uploaded person photo and apply saree on that person" if has_user_photo else "Use an Indian female model"}
"""

        # -------- BUILD MULTIPART (CORRECT FORMAT) --------
        files = [
            ("model", (None, "gpt-image-1")),
            ("prompt", (None, prompt)),
            ("size", (None, "1024x1024")),
        ]

        # ADD IMAGES (IMPORTANT)
        files.append(("image", ("pallu.jpg", pallu_bytes, "image/jpeg")))
        files.append(("image", ("blouse.jpg", blouse_bytes, "image/jpeg")))
        files.append(("image", ("fabric.jpg", fabric_bytes, "image/jpeg")))

        if has_user_photo:
            files.append(("image", ("user.jpg", user_bytes, "image/jpeg")))

        print("Calling OpenAI API...")

        async with httpx.AsyncClient(timeout=TIMEOUT) as client:
            response = await client.post(
                "https://api.openai.com/v1/images/edits",
                headers={
                    "Authorization": f"Bearer {api_key}"
                },
                files=files,
            )

        print("STATUS:", response.status_code)
        print("RESPONSE:", response.text[:500])  # avoid huge logs

        if response.status_code != 200:
            return JSONResponse(
                status_code=500,
                content={
                    "error": "OpenAI API failed",
                    "status_code": response.status_code,
                    "details": response.text
                }
            )

        result = response.json()

        if not result.get("data"):
            return JSONResponse(
                status_code=500,
                content={
                    "error": "No image returned",
                    "response": result
                }
            )

        image_b64 = result["data"][0].get("b64_json")

        if not image_b64:
            return JSONResponse(
                status_code=500,
                content={
                    "error": "Invalid image response",
                    "response": result
                }
            )

        return JSONResponse(content={
            "success": True,
            "image_base64": image_b64,
            "used_images": True
        })

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