import os
import base64
import traceback
import httpx
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
    api_key = os.environ.get("OPENAI_API_KEY", "")
    return {
        "status": "Saree AI API is running",
        "openai_key_set": bool(api_key),
        "key_length": len(api_key)
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
        api_key = os.environ.get("OPENAI_API_KEY", "")
        if not api_key:
            return JSONResponse(
                status_code=500,
                content={"error": "OPENAI_API_KEY is not set in environment variables."}
            )

        # Read uploaded images and convert to base64 for GPT-4o Vision
        pallu_bytes = await pallu_image.read()
        blouse_bytes = await blouse_image.read()
        fabric_bytes = await fabric_image.read()

        pallu_b64 = base64.b64encode(pallu_bytes).decode("utf-8")
        blouse_b64 = base64.b64encode(blouse_bytes).decode("utf-8")
        fabric_b64 = base64.b64encode(fabric_bytes).decode("utf-8")

        pallu_mime = pallu_image.content_type or "image/jpeg"
        blouse_mime = blouse_image.content_type or "image/jpeg"
        fabric_mime = fabric_image.content_type or "image/jpeg"

        # Build prompt
        prompt = SAREE_PROMPT.replace("{drape_style}", drape_style.capitalize())

        # Step 1: Use GPT-4o Vision to analyze the uploaded images
        # and generate a detailed DALL-E prompt from them
        vision_messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": f"""You are a fashion image prompt engineer. 
Analyze these saree fabric images carefully and create a highly detailed DALL-E 3 image generation prompt.

The prompt must:
1. Describe the exact colors, patterns, motifs, borders seen in the fabric images
2. Describe the blouse color, style, neckline, sleeve type
3. Describe the pallu design and border details
4. Apply {drape_style} draping style
5. Include: elegant Indian woman, traditional juda bun with white gajra, minimal makeup, gold jewelry
6. Photography: premium fashion catalog, studio lighting, beige background, full body visible
7. Output: 3-angle composite (front, 45-degree side, back views)

Return ONLY the DALL-E prompt text, nothing else. Make it extremely detailed and specific about the fabric patterns and colors you see."""
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:{pallu_mime};base64,{pallu_b64}",
                            "detail": "high"
                        }
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:{blouse_mime};base64,{blouse_b64}",
                            "detail": "high"
                        }
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:{fabric_mime};base64,{fabric_b64}",
                            "detail": "high"
                        }
                    },
                ]
            }
        ]

        # Add user photo if provided
        if user_photo and user_photo.filename:
            user_bytes = await user_photo.read()
            if user_bytes:
                user_b64 = base64.b64encode(user_bytes).decode("utf-8")
                user_mime = user_photo.content_type or "image/jpeg"
                vision_messages[0]["content"].append({
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:{user_mime};base64,{user_b64}",
                        "detail": "high"
                    }
                })
                vision_messages[0]["content"][0]["text"] += "\n\nA user photo is also provided. Mention in the prompt to dress this specific person in the saree while preserving their face and body proportions."

        async with httpx.AsyncClient(timeout=120.0) as client:

            # Step 1: Analyze images with GPT-4o and get detailed DALL-E prompt
            vision_response = await client.post(
                "https://api.openai.com/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": "gpt-4o",
                    "messages": vision_messages,
                    "max_tokens": 1000,
                }
            )

            vision_result = vision_response.json()

            if "error" in vision_result:
                return JSONResponse(
                    status_code=500,
                    content={
                        "error": f"GPT-4o Vision error: {vision_result['error'].get('message', 'Unknown error')}",
                    }
                )

            dalle_prompt = vision_result["choices"][0]["message"]["content"].strip()
            print(f"Generated DALL-E prompt: {dalle_prompt[:200]}...")

            # Step 2: Generate image with DALL-E 3 using the detailed prompt
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
                    "response_format": "b64_json",
                    "quality": "hd",
                    "style": "natural",
                }
            )

            image_result = image_response.json()

            if "error" in image_result:
                return JSONResponse(
                    status_code=500,
                    content={
                        "error": f"DALL-E 3 error: {image_result['error'].get('message', 'Unknown error')}",
                        "dalle_prompt_used": dalle_prompt
                    }
                )

            image_b64 = image_result["data"][0]["b64_json"]
            revised_prompt = image_result["data"][0].get("revised_prompt", dalle_prompt)

        return JSONResponse(content={
            "success": True,
            "image_base64": image_b64,
            "mime_type": "image/png",
            "message": revised_prompt,
            "model_used": "gpt-4o + dall-e-3",
            "dalle_prompt": dalle_prompt
        })

    except Exception as e:
        detail = traceback.format_exc()
        print("ERROR:\n", detail)
        return JSONResponse(
            status_code=500,
            content={"error": str(e), "traceback": detail}
        )