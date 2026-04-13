import os
import base64
import traceback
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from typing import Optional
from dotenv import load_dotenv
import google.generativeai as genai

load_dotenv()

app = FastAPI(title="Saree AI API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

GEMINI_API_KEY = os.getenv("API_KEY")
genai.configure(api_key=GEMINI_API_KEY)

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
Draping style: {drape_style} saree drape
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


@app.get("/")
def root():
    return {
        "status": "Saree AI API is running",
        "gemini_key_set": bool(GEMINI_API_KEY)
    }


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/generate-saree")
async def generate_saree(
    pallu_image: UploadFile = File(...),
    blouse_image: UploadFile = File(...),
    fabric_image: UploadFile = File(...),
    user_photo: Optional[UploadFile] = File(None),
    drape_style: str = Form("Bengali"),
):
    try:
        # Read all uploaded images
        pallu_bytes = await pallu_image.read()
        blouse_bytes = await blouse_image.read()
        fabric_bytes = await fabric_image.read()

        pallu_mime = pallu_image.content_type or "image/jpeg"
        blouse_mime = blouse_image.content_type or "image/jpeg"
        fabric_mime = fabric_image.content_type or "image/jpeg"

        # Build the prompt
        prompt = SAREE_PROMPT.replace("{drape_style}", drape_style.capitalize())

        # Build content list for Gemini
        content = []

        content.append("PALLU IMAGE:")
        content.append({"mime_type": pallu_mime, "data": pallu_bytes})

        content.append("BLOUSE IMAGE:")
        content.append({"mime_type": blouse_mime, "data": blouse_bytes})

        content.append("MAIN SAREE FABRIC IMAGE:")
        content.append({"mime_type": fabric_mime, "data": fabric_bytes})

        if user_photo and user_photo.filename:
            user_bytes = await user_photo.read()
            if user_bytes:
                user_mime = user_photo.content_type or "image/jpeg"
                content.append("USER FULL-BODY PHOTO (TRY-ON MODE - preserve face and body):")
                content.append({"mime_type": user_mime, "data": user_bytes})

        content.append(prompt)

        # Use the Gemini image generation model
        model = genai.GenerativeModel(model_name="gemini-2.0-flash-preview-image-generation")

        response = model.generate_content(
            content,
            generation_config=genai.GenerationConfig(
                response_modalities=["TEXT", "IMAGE"]
            )
        )

        # Extract image from response
        generated_image_b64 = None
        text_response = ""

        if not response.candidates:
            return JSONResponse(
                status_code=500,
                content={"error": "No candidates in response", "details": str(response)}
            )

        candidate = response.candidates[0]

        for part in candidate.content.parts:
            if hasattr(part, "inline_data") and part.inline_data is not None:
                data = part.inline_data.data
                if isinstance(data, bytes):
                    generated_image_b64 = base64.b64encode(data).decode("utf-8")
                elif isinstance(data, str):
                    generated_image_b64 = data  # already base64
            elif hasattr(part, "text") and part.text:
                text_response += part.text

        if not generated_image_b64:
            return JSONResponse(
                status_code=500,
                content={
                    "error": "No image was generated",
                    "text_response": text_response,
                    "finish_reason": str(candidate.finish_reason) if hasattr(candidate, "finish_reason") else "unknown"
                }
            )

        return JSONResponse(content={
            "success": True,
            "image_base64": generated_image_b64,
            "mime_type": "image/png",
            "message": text_response
        })

    except Exception as e:
        error_detail = traceback.format_exc()
        print("ERROR:", error_detail)
        return JSONResponse(
            status_code=500,
            content={
                "error": str(e),
                "traceback": error_detail
            }
        )