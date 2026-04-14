import os
import base64
import traceback
import httpx
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

PHOTOGRAPHY:
- Premium fashion catalog shoot
- Studio lighting with soft shadows
- Neutral beige background
- Sharp focus on fabric and draping
- Ultra-realistic quality
- Full body visible"""


@app.get("/")
def root():
    api_key = os.environ.get("OPENAI_API_KEY", "")
    return {
        "status": "Saree AI API is running",
        "openai_key_set": bool(api_key),
        "key_length": len(api_key),
    }


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/generate-saree")
async def generate_saree(
    prompt: str = Form(None),
    drape_style: str = Form("Bengali"),
):
    try:
        api_key = os.environ.get("OPENAI_API_KEY", "")
        if not api_key:
            return JSONResponse(
                status_code=500,
                content={"error": "OPENAI_API_KEY is not set in environment variables."}
            )

        # Build final prompt
        final_prompt = prompt.strip() if (prompt and prompt.strip()) else DEFAULT_PROMPT
        final_prompt = final_prompt.replace("Bengali drape style", f"{drape_style} drape style")
        final_prompt = final_prompt.replace("Bengali saree drape (default)", f"{drape_style} saree drape")

        # Call OpenAI DALL-E 3
        async with httpx.AsyncClient(timeout=120.0) as client:
            response = await client.post(
                "https://api.openai.com/v1/images/generations",
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": "dall-e-3",
                    "prompt": final_prompt,
                    "n": 1,
                    "size": "1024x1024",
                    "response_format": "b64_json",
                    "quality": "hd",
                    "style": "natural",
                }
            )

        result = response.json()

        # Check for API errors
        if "error" in result:
            return JSONResponse(
                status_code=500,
                content={
                    "error": result["error"].get("message", "OpenAI API error"),
                    "code": result["error"].get("code", "unknown")
                }
            )

        # Extract base64 image
        image_b64 = result["data"][0]["b64_json"]
        revised_prompt = result["data"][0].get("revised_prompt", "")

        return JSONResponse(content={
            "success": True,
            "image_base64": image_b64,
            "mime_type": "image/png",
            "message": revised_prompt,
            "model_used": "dall-e-3"
        })

    except Exception as e:
        detail = traceback.format_exc()
        print("ERROR:\n", detail)
        return JSONResponse(
            status_code=500,
            content={"error": str(e), "traceback": detail}
        )