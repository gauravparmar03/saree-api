import os
import base64
import traceback
from fastapi import FastAPI, Form
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="Saree AI API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

DEFAULT_PROMPT = """Create a highly realistic, full-body fashion image of a saree.

MODEL STYLING:
- elegant Indian woman
- hair in traditional juda bun with white gajra
- minimal natural makeup
- traditional gold jewelry: necklace, jhumkas, bangles

SAREE DRAPING:
- Bengali drape style
- wide front pleats
- pallu over left shoulder
- visible border and tassels
- realistic fabric physics and folds

COLORS AND FABRIC:
- rich silk saree with golden zari border
- deep red or maroon base color
- intricate traditional motifs

OUTPUT:
- Single composite image with 3 angles: front, 45-degree side, back
- Full body visible in all views
- Same person across all angles

PHOTOGRAPHY:
- Premium fashion catalog shoot
- Studio lighting with soft shadows
- Neutral beige background
- Sharp focus on fabric and draping
- Ultra-realistic quality"""


@app.get("/")
def root():
    api_key = os.environ.get("API_KEY")
    return {"status": "Saree AI API is running", "gemini_key_set": bool(api_key)}


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/generate-saree")
async def generate_saree(
    prompt: str = Form(None),
    drape_style: str = Form("Bengali"),
):
    try:
        # Read API key fresh on every request (works on Render)
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            return JSONResponse(
                status_code=500,
                content={"error": "GEMINI_API_KEY is not set in environment variables"}
            )

        # Import here so client is created with fresh key
        from google import genai
        from google.genai import types

        gemini_client = genai.Client(api_key=api_key)

        # Build prompt
        final_prompt = prompt.strip() if (prompt and prompt.strip()) else DEFAULT_PROMPT
        final_prompt = final_prompt.replace("Bengali drape style", f"{drape_style} drape style")
        final_prompt = final_prompt.replace("Bengali saree drape (default)", f"{drape_style} saree drape")

        response = gemini_client.models.generate_content(
            model="gemini-2.0-flash-exp",
            contents=final_prompt,
            config=types.GenerateContentConfig(
                response_modalities=["Text", "Image"]
            )
        )

        if not response.candidates:
            return JSONResponse(
                status_code=500,
                content={"error": "No candidates returned by Gemini"}
            )

        candidate = response.candidates[0]
        generated_image_b64 = None
        text_response = ""

        for part in candidate.content.parts:
            if part.inline_data is not None:
                data = part.inline_data.data
                if isinstance(data, bytes):
                    generated_image_b64 = base64.b64encode(data).decode("utf-8")
                elif isinstance(data, str):
                    generated_image_b64 = data
            elif part.text:
                text_response += part.text

        if not generated_image_b64:
            return JSONResponse(
                status_code=500,
                content={
                    "error": "Gemini did not return an image",
                    "text_response": text_response,
                    "finish_reason": str(getattr(candidate, "finish_reason", "unknown"))
                }
            )

        return JSONResponse(content={
            "success": True,
            "image_base64": generated_image_b64,
            "mime_type": "image/png",
            "message": text_response
        })

    except Exception as e:
        detail = traceback.format_exc()
        print("ERROR:\n", detail)
        return JSONResponse(
            status_code=500,
            content={"error": str(e), "traceback": detail}
        )