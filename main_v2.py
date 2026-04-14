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

MODELS_TO_TRY = [
    "gemini-2.5-flash-image-preview",
    "gemini-3.1-flash-image-preview",
    "gemini-2.5-flash-preview-04-17",
]


@app.get("/")
def root():
    api_key = os.environ.get("GEMINI_API_KEY", "")
    return {
        "status": "Saree AI API is running",
        "gemini_key_set": bool(api_key),
        "key_length": len(api_key),
    }


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/list-models")
def list_models():
    try:
        api_key = os.environ.get("GEMINI_API_KEY", "")
        from google import genai
        gemini_client = genai.Client(api_key=api_key)
        models = gemini_client.models.list()
        all_models = [m.name for m in models]
        image_models = [m for m in all_models if any(
            x in m.lower() for x in ["image", "flash", "imagen", "pro"]
        )]
        return {"image_related_models": image_models, "all_models": all_models}
    except Exception as e:
        return {"error": str(e), "traceback": traceback.format_exc()}


@app.post("/generate-saree")
async def generate_saree(
    prompt: str = Form(None),
    drape_style: str = Form("Bengali"),
):
    try:
        api_key = os.environ.get("GEMINI_API_KEY", "")
        if not api_key:
            return JSONResponse(
                status_code=500,
                content={"error": "GEMINI_API_KEY is not set."}
            )

        from google import genai
        from google.genai import types

        gemini_client = genai.Client(api_key=api_key)

        final_prompt = prompt.strip() if (prompt and prompt.strip()) else DEFAULT_PROMPT
        final_prompt = final_prompt.replace("Bengali drape style", f"{drape_style} drape style")
        final_prompt = final_prompt.replace("Bengali saree drape (default)", f"{drape_style} saree drape")

        last_error = None
        response = None
        working_model = None

        for model_name in MODELS_TO_TRY:
            try:
                print(f"Trying model: {model_name}")
                response = gemini_client.models.generate_content(
                    model=model_name,
                    contents=final_prompt,
                    config=types.GenerateContentConfig(
                        response_modalities=["Text", "Image"]
                    )
                )
                working_model = model_name
                print(f"Success with model: {model_name}")
                break
            except Exception as model_err:
                last_error = str(model_err)
                print(f"Model {model_name} failed: {model_err}")
                continue

        if response is None:
            return JSONResponse(
                status_code=500,
                content={
                    "error": "All models failed. Check /list-models to see available models.",
                    "last_error": last_error,
                    "models_tried": MODELS_TO_TRY
                }
            )

        if not response.candidates:
            return JSONResponse(
                status_code=500,
                content={"error": "No candidates returned", "model_used": working_model}
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
                    "finish_reason": str(getattr(candidate, "finish_reason", "unknown")),
                    "model_used": working_model
                }
            )

        return JSONResponse(content={
            "success": True,
            "image_base64": generated_image_b64,
            "mime_type": "image/png",
            "message": text_response,
            "model_used": working_model
        })

    except Exception as e:
        detail = traceback.format_exc()
        print("ERROR:\n", detail)
        return JSONResponse(
            status_code=500,
            content={"error": str(e), "traceback": detail}
        )