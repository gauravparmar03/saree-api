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

        # Read all images
        pallu_bytes = await pallu_image.read()
        blouse_bytes = await blouse_image.read()
        fabric_bytes = await fabric_image.read()

        has_user_photo = False
        user_bytes = None
        if user_photo and user_photo.filename:
            user_bytes = await user_photo.read()
            if user_bytes:
                has_user_photo = True

        # --- Step 1: GPT-4o Vision analyzes all images in detail ---
        vision_content = [
            {
                "type": "text",
                "text": f"""You are an expert fashion analyst and prompt engineer specializing in Indian ethnic wear.

Analyze the provided saree fabric images VERY carefully and extract every detail:

FROM PALLU IMAGE: Describe exact border design, motifs, colors, patterns, tassel details, zari work
FROM BLOUSE IMAGE: Describe exact color, neckline shape, sleeve length/style, embroidery, fit
FROM FABRIC IMAGE: Describe exact base color, weave pattern, motifs, texture, border stripes

{"FROM USER PHOTO: Describe the person's skin tone, body proportions, height, face features to preserve." if has_user_photo else ""}

Then write a DALL-E 3 image generation prompt that:

1. FABRIC ACCURACY: Uses exact colors/patterns/motifs you described from the images
2. DRAPING: {drape_style} saree drape style - wide front pleats, pallu over left shoulder, visible border and tassels, realistic fabric physics
3. BLOUSE: Exactly matches the blouse image details
4. {"PERSON: Dress the specific person from the user photo in this saree. Preserve their exact face, skin tone, body proportions. The saree replaces their existing clothing naturally." if has_user_photo else "MODEL: Elegant Indian woman, traditional juda bun with white gajra flowers, minimal natural makeup, traditional gold jewelry (necklace, jhumkas, bangles)"}
5. OUTPUT: THREE-ANGLE composite image - LEFT shows front view full body, CENTER shows 45-degree side view, RIGHT shows back view showing pallu fall clearly
6. PHOTOGRAPHY: Premium fashion catalog shoot, professional studio lighting with soft shadows, neutral beige/cream background, sharp focus throughout, full body visible in all three views
7. QUALITY: Ultra-realistic, 4K quality, accurate fabric draping physics, no pattern distortion

Return ONLY the prompt text. Be extremely specific about colors and patterns."""
            },
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:{pallu_image.content_type or 'image/jpeg'};base64,{base64.b64encode(pallu_bytes).decode()}",
                    "detail": "high"
                }
            },
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:{blouse_image.content_type or 'image/jpeg'};base64,{base64.b64encode(blouse_bytes).decode()}",
                    "detail": "high"
                }
            },
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:{fabric_image.content_type or 'image/jpeg'};base64,{base64.b64encode(fabric_bytes).decode()}",
                    "detail": "high"
                }
            }
        ]

        # Add user photo to vision analysis if provided
        if has_user_photo:
            vision_content.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:{user_photo.content_type or 'image/jpeg'};base64,{base64.b64encode(user_bytes).decode()}",
                    "detail": "high"
                }
            })

        async with httpx.AsyncClient(timeout=180.0) as client:

            # Step 1: Get detailed prompt from GPT-4o Vision
            print("Step 1: Analyzing images with GPT-4o Vision...")
            vision_response = await client.post(
                "https://api.openai.com/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": "gpt-4o",
                    "messages": [{"role": "user", "content": vision_content}],
                    "max_tokens": 1500,
                }
            )

            vision_result = vision_response.json()
            if "error" in vision_result:
                return JSONResponse(
                    status_code=500,
                    content={"error": f"GPT-4o error: {vision_result['error'].get('message')}"}
                )

            dalle_prompt = vision_result["choices"][0]["message"]["content"].strip()
            print(f"Generated prompt ({len(dalle_prompt)} chars): {dalle_prompt[:300]}...")

            # Step 2: Try gpt-image-1 first (takes real image input), fallback to DALL-E 3
            image_b64 = None
            model_used = None
            revised_prompt = dalle_prompt

            # Try gpt-image-1 with actual fabric images as reference
            try:
                print("Step 2a: Trying gpt-image-1 with image references...")

                # Build multipart form for gpt-image-1
                import io
                files = {
                    "model": (None, "gpt-image-1"),
                    "prompt": (None, dalle_prompt),
                    "n": (None, "1"),
                    "size": (None, "1024x1024"),
                    "quality": (None, "high"),
                }

                # Attach fabric images as references
                files["image[]"] = ("pallu.jpg", pallu_bytes, pallu_image.content_type or "image/jpeg")

                img1_response = await client.post(
                    "https://api.openai.com/v1/images/generations",
                    headers={"Authorization": f"Bearer {api_key}"},
                    files=files,
                )
                img1_result = img1_response.json()

                if "error" not in img1_result and img1_result.get("data"):
                    data = img1_result["data"][0]
                    if data.get("b64_json"):
                        image_b64 = data["b64_json"]
                    elif data.get("url"):
                        # Download the image from URL
                        img_download = await client.get(data["url"])
                        image_b64 = base64.b64encode(img_download.content).decode("utf-8")
                    model_used = "gpt-image-1"
                    print("gpt-image-1 succeeded!")
                else:
                    print(f"gpt-image-1 failed: {img1_result.get('error', {}).get('message', 'unknown')}")

            except Exception as e1:
                print(f"gpt-image-1 exception: {e1}")

            # Fallback to DALL-E 3 if gpt-image-1 failed
            if not image_b64:
                print("Step 2b: Falling back to DALL-E 3...")
                dalle_response = await client.post(
                    "https://api.openai.com/v1/images/generations",
                    headers={
                        "Authorization": f"Bearer {api_key}",
                        "Content-Type": "application/json",
                    },
                    json={
                        "model": "dall-e-3",
                        "prompt": dalle_prompt,
                        "n": 1,
                        "size": "1792x1024",  # wider for 3-angle composite
                        "response_format": "b64_json",
                        "quality": "hd",
                        "style": "natural",
                    }
                )

                dalle_result = dalle_response.json()
                if "error" in dalle_result:
                    return JSONResponse(
                        status_code=500,
                        content={
                            "error": f"DALL-E 3 error: {dalle_result['error'].get('message')}",
                            "dalle_prompt": dalle_prompt
                        }
                    )

                image_b64 = dalle_result["data"][0]["b64_json"]
                revised_prompt = dalle_result["data"][0].get("revised_prompt", dalle_prompt)
                model_used = "dall-e-3"
                print("DALL-E 3 succeeded!")

        return JSONResponse(content={
            "success": True,
            "image_base64": image_b64,
            "mime_type": "image/png",
            "message": revised_prompt,
            "model_used": model_used,
            "dalle_prompt": dalle_prompt
        })

    except Exception as e:
        detail = traceback.format_exc()
        print("ERROR:\n", detail)
        return JSONResponse(
            status_code=500,
            content={"error": str(e), "traceback": detail}
        )