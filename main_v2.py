import os
import base64
import traceback
from fastapi import FastAPI, Form
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
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

OUTPUT:
- Single composite image with 3 angles: front, 45-degree side, back
- Full body visible in all views
- Same person across all angles

PHOTOGRAPHY:
- Premium fashion catalog shoot
- Studio lighting with soft shadows
- Neutral beige background
- Sharp focus on fabric and draping
- Ultra-realistic 4K quality"""


@app.get("/")
def root():
    return {"status": "Saree AI API is running", "gemini_key_set": bool(GEMINI_API_KEY)}


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/generate-saree")
async def generate_saree(
    prompt: str = Form(None),
    drape_style: str = Form("Bengali"),
):
    try:
        final_prompt = prompt if (prompt and prompt.strip()) else DEFAULT_PROMPT
        final_prompt = final_prompt.replace("Bengali drape style", f"{drape_style} drape style")
        final_prompt = final_prompt.replace("Bengali saree drape (default)", f"{drape_style} saree drape")

        model = genai.GenerativeModel(model_name="gemini-2.0-flash-preview-image-generation")

        response = model.generate_content(
            final_prompt,
            generation_config=genai.GenerationConfig(
                response_modalities=["TEXT", "IMAGE"]
            )
        )

        if not response.candidates:
            return JSONResponse(
                status_code=500,
                content={"error": "No candidates returned by Gemini", "raw": str(response)}
            )

        candidate = response.candidates[0]
        generated_image_b64 = None
        text_response = ""

        for part in candidate.content.parts:
            if hasattr(part, "inline_data") and part.inline_data is not None:
                data = part.inline_data.data
                if isinstance(data, bytes):
                    generated_image_b64 = base64.b64encode(data).decode("utf-8")
                elif isinstance(data, str):
                    generated_image_b64 = data
            elif hasattr(part, "text") and part.text:
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
