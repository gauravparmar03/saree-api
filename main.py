import os
import base64
import traceback
import httpx
import io
from PIL import Image
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from typing import Optional

app = FastAPI(title="Saree AI API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------- CONFIG ----------
MAX_SIZE = 2 * 1024 * 1024  # 2MB
TIMEOUT = 60.0

# ---------- UTIL ----------
def compress_image(image_bytes):
    try:
        img = Image.open(io.BytesIO(image_bytes))
        img = img.convert("RGB")

        buffer = io.BytesIO()
        img.save(buffer, format="JPEG", quality=70)
        return buffer.getvalue()
    except Exception:
        return image_bytes  # fallback


async def read_and_validate(file: UploadFile, name: str):
    if not file or not file.filename:
        raise Exception(f"{name} is missing")

    content = await file.read()

    if not content:
        raise Exception(f"{name} is empty")

    if len(content) > MAX_SIZE:
        raise Exception(f"{name} too large (max 2MB)")

    return compress_image(content)


# ---------- HEALTH ----------
@app.get("/")
def root():
    api_key = os.environ.get("OPENAI_API_KEY", "")
    return {
        "status": "API running",
        "openai_key_set": bool(api_key),
        "key_length": len(api_key)
    }


@app.get("/health")
def health():
    return {"status": "ok"}


# ---------- MAIN API ----------
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

        # ---------- READ FILES ----------
        pallu_bytes = await read_and_validate(pallu_image, "pallu_image")
        blouse_bytes = await read_and_validate(blouse_image, "blouse_image")
        fabric_bytes = await read_and_validate(fabric_image, "fabric_image")

        user_bytes = None
        has_user_photo = False

        if user_photo and user_photo.filename:
            user_bytes = await read_and_validate(user_photo, "user_photo")
            has_user_photo = True

        print("Files received successfully")

        # ---------- BUILD VISION INPUT ----------
        vision_content = [
            {
                "type": "text",
                "text": f"""
Analyze saree images and create a DALL-E prompt.

- Extract colors, patterns, motifs
- Use {drape_style} drape
- Create ultra realistic fashion output
{"- Use provided person photo exactly" if has_user_photo else "- Use Indian model"}
Return only prompt text.
"""
            },
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{base64.b64encode(pallu_bytes).decode()}",
                    "detail": "high"
                }
            },
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{base64.b64encode(blouse_bytes).decode()}",
                    "detail": "high"
                }
            },
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{base64.b64encode(fabric_bytes).decode()}",
                    "detail": "high"
                }
            }
        ]

        if has_user_photo:
            vision_content.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{base64.b64encode(user_bytes).decode()}",
                    "detail": "high"
                }
            })

        # ---------- CALL OPENAI ----------
        async with httpx.AsyncClient(timeout=TIMEOUT) as client:

            # ---- STEP 1: VISION ----
            print("Calling Vision API...")
            vision_response = await client.post(
                "https://api.openai.com/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": "gpt-4o",
                    "messages": [{"role": "user", "content": vision_content}],
                    "max_tokens": 1000,
                }
            )

            if vision_response.status_code != 200:
                return JSONResponse(
                    status_code=500,
                    content={
                        "error": "Vision API failed",
                        "details": vision_response.text
                    }
                )

            vision_result = vision_response.json()
            print("Vision response received")

            try:
                dalle_prompt = vision_result["choices"][0]["message"]["content"].strip()
            except Exception:
                return JSONResponse(
                    status_code=500,
                    content={
                        "error": "Invalid OpenAI response",
                        "response": vision_result
                    }
                )

            # ---- STEP 2: IMAGE GENERATION ----
            print("Generating image...")

            image_response = await client.post(
                "https://api.openai.com/v1/images/generations",
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": "dall-e-3",
                    "prompt": dalle_prompt,
                    "n": 1,
                    "size": "1024x1024",
                    "response_format": "b64_json"
                }
            )

            if image_response.status_code != 200:
                return JSONResponse(
                    status_code=500,
                    content={
                        "error": "Image generation failed",
                        "details": image_response.text
                    }
                )

            image_result = image_response.json()

            try:
                image_b64 = image_result["data"][0]["b64_json"]
            except Exception:
                return JSONResponse(
                    status_code=500,
                    content={
                        "error": "Invalid image response",
                        "response": image_result
                    }
                )

        return JSONResponse(content={
            "success": True,
            "image_base64": image_b64,
            "prompt": dalle_prompt
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