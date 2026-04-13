import os
import base64
import io
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import google.generativeai as genai
from dotenv import load_dotenv
from typing import Optional

load_dotenv()

app = FastAPI(title="Saree AI API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

genai.configure(api_key=os.getenv("API_KEY"))

SAREE_PROMPT = """Create a highly realistic, full-body fashion image of a saree applied using the provided input images:
1) saree pallu image
2) blouse image
3) main saree fabric image
4) (optional) user full-body photo for try-on

PRIMARY REQUIREMENT:
Use the provided images as direct texture and design source. Perform strict fabric mapping without redesigning, approximating, or inventing patterns.

FABRIC ACCURACY (CRITICAL):
- Preserve exact colors, gradients, and weave texture
- Maintain exact motif shapes, sizes, spacing, and orientation
- Ensure border design, stripe placement, and thickness are identical
- Pallu must show correct design, border alignment, and tassels
- Do not distort, blur, or reinterpret fabric patterns

DRAPING:
Apply realistic saree draping with correct fabric physics and folds.
Draping style: Bengali saree drape (default)
- wide front pleats
- pallu over left shoulder
- visible border and tassels

BLOUSE:
- Must exactly match blouse image (color, sleeves, neckline, fit)

IF USER PHOTO PROVIDED (TRY-ON MODE):
- Preserve exact face identity and body proportions
- Do not modify facial features
- Replace existing clothing naturally with saree
- Maintain realistic lighting and skin tone consistency

MODEL STYLING:
- elegant Indian woman
- hair in traditional juda (bun) with white gajra
- minimal natural makeup
- traditional gold jewelry (necklace, jhumkas, bangles)

OUTPUT FORMAT:
- Generate a single composite image showing 3 angles:
  1) front view
  2) side view (45-degree)
  3) back view (clear pallu fall)
- Ensure same person and same saree consistency across all angles
- Full-body visible in all views

PHOTOGRAPHY STYLE:
- premium fashion catalog shoot
- studio lighting with soft shadows
- neutral beige or pastel background
- sharp focus on fabric and draping

QUALITY CONSTRAINTS:
- ultra-realistic (4K quality)
- accurate saree pleats and fall
- no fabric distortion
- no extra or missing patterns
- no hallucinated elements
- consistent identity across views

STRICT MODE:
Prioritize textile accuracy over creativity. Do not modify or enhance the design. Reproduce exactly from input images."""


def pil_to_part(img_bytes: bytes, mime_type: str = "image/jpeg"):
    return {"mime_type": mime_type, "data": img_bytes}


@app.get("/")
def root():
    return {"status": "Saree AI API is running"}


@app.post("/generate-saree")
async def generate_saree(
    pallu_image: UploadFile = File(...),
    blouse_image: UploadFile = File(...),
    fabric_image: UploadFile = File(...),
    user_photo: Optional[UploadFile] = File(None),
    drape_style: str = Form("bengali"),
):
    try:
        # Read uploaded files
        pallu_bytes = await pallu_image.read()
        blouse_bytes = await blouse_image.read()
        fabric_bytes = await fabric_image.read()

        # Build prompt with drape style
        prompt = SAREE_PROMPT.replace(
            "Draping style: Bengali saree drape (default)",
            f"Draping style: {drape_style.capitalize()} saree drape"
        )

        # Build content parts
        parts = [
            {"text": "PALLU IMAGE:"},
            pil_to_part(pallu_bytes, pallu_image.content_type or "image/jpeg"),
            {"text": "BLOUSE IMAGE:"},
            pil_to_part(blouse_bytes, blouse_image.content_type or "image/jpeg"),
            {"text": "MAIN SAREE FABRIC IMAGE:"},
            pil_to_part(fabric_bytes, fabric_image.content_type or "image/jpeg"),
        ]

        if user_photo:
            user_bytes = await user_photo.read()
            parts.append({"text": "USER FULL-BODY PHOTO (TRY-ON MODE):"})
            parts.append(pil_to_part(user_bytes, user_photo.content_type or "image/jpeg"))

        parts.append({"text": prompt})

        # Call Gemini image generation model
        model = genai.GenerativeModel("gemini-2.0-flash-preview-image-generation")
        response = model.generate_content(
            parts,
            generation_config=genai.GenerationConfig(
                response_modalities=["TEXT", "IMAGE"],
            )
        )

        # Extract generated image
        generated_image_b64 = None
        text_response = ""

        for part in response.candidates[0].content.parts:
            if hasattr(part, "inline_data") and part.inline_data:
                image_data = part.inline_data.data
                if isinstance(image_data, bytes):
                    generated_image_b64 = base64.b64encode(image_data).decode("utf-8")
                else:
                    generated_image_b64 = image_data  # already b64 string
            elif hasattr(part, "text") and part.text:
                text_response = part.text

        if not generated_image_b64:
            return JSONResponse(
                status_code=500,
                content={"error": "No image generated", "details": text_response}
            )

        return JSONResponse(content={
            "success": True,
            "image_base64": generated_image_b64,
            "mime_type": "image/png",
            "message": text_response
        })

    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": str(e)}
        )