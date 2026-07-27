import os
import base64
import traceback
import httpx
import io
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from typing import Optional
from PIL import Image

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


def make_square_rgba_png(image_bytes: bytes, size: int = 512) -> bytes:
    """Convert image bytes to a square RGBA PNG at the given size."""
    img = Image.open(io.BytesIO(image_bytes)).convert("RGBA")
    img = img.resize((size, size), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def composite_images_grid(images_bytes: list[bytes], cell_size: int = 512) -> bytes:
    """
    Arrange images in a horizontal strip, each resized to cell_size x cell_size.
    Returns PNG bytes of the composite.
    """
    n = len(images_bytes)
    composite = Image.new("RGBA", (cell_size * n, cell_size), (255, 255, 255, 255))
    for i, b in enumerate(images_bytes):
        img = Image.open(io.BytesIO(b)).convert("RGBA")
        img = img.resize((cell_size, cell_size), Image.LANCZOS)
        composite.paste(img, (i * cell_size, 0))
    buf = io.BytesIO()
    composite.save(buf, format="PNG")
    return buf.getvalue()


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

        # ---------------------------------------------------------------
        # Step 1: GPT-4o Vision analyzes all images and writes a prompt
        # ---------------------------------------------------------------
        vision_content = [
            {
                "type": "text",
                "text": f"""You are an expert fashion analyst and prompt engineer specializing in Indian ethnic wear.

Analyze the provided saree fabric images VERY carefully and extract every detail:

FROM PALLU IMAGE (first image): Describe exact border design, motifs, colors, patterns, tassel details, zari work
FROM BLOUSE IMAGE (second image): Describe exact color, neckline shape, sleeve length/style, embroidery, fit
FROM FABRIC IMAGE (third image): Describe exact base color, weave pattern, motifs, texture, border stripes

{"FROM USER PHOTO (fourth image): Describe the person's skin tone, body proportions, height, face features to preserve." if has_user_photo else ""}

Then write an image generation prompt that:

1. FABRIC ACCURACY: Uses exact colors/patterns/motifs you described from the images
2. DRAPING: {drape_style} saree drape style — wide front pleats, pallu over left shoulder, visible border and tassels, realistic fabric physics
3. BLOUSE: Exactly matches the blouse image details
4. {"PERSON: Dress the specific person from the user photo in this saree. Preserve their exact face, skin tone, body proportions. The saree replaces their existing clothing naturally." if has_user_photo else "MODEL: Elegant Indian woman, traditional juda bun with white gajra flowers, minimal natural makeup, traditional gold jewelry (necklace, jhumkas, bangles)"}
5. OUTPUT: THREE-ANGLE composite image — LEFT shows front view full body, CENTER shows 45-degree side view, RIGHT shows back view showing pallu fall clearly
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

        if has_user_photo:
            vision_content.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:{user_photo.content_type or 'image/jpeg'};base64,{base64.b64encode(user_bytes).decode()}",
                    "detail": "high"
                }
            })

        async with httpx.AsyncClient(timeout=180.0) as client:

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

            # ---------------------------------------------------------------
            # Step 2a: gpt-image-1 via /images/edits — passes ACTUAL images
            # ---------------------------------------------------------------
            # gpt-image-1 supports real image references through the edits endpoint.
            # We composite the fabric reference images (pallu + fabric + blouse) into
            # a single horizontal strip so the model can directly "see" the materials.
            # If a user photo is supplied, it becomes the base image being edited.
            # ---------------------------------------------------------------
            image_b64 = None
            model_used = None
            revised_prompt = dalle_prompt

            try:
                print("Step 2a: Trying gpt-image-1 with real image references (/images/edits)...")

                # Build the reference composite: pallu | fabric | blouse
                ref_images = [pallu_bytes, fabric_bytes, blouse_bytes]
                reference_composite = composite_images_grid(ref_images, cell_size=512)

                if has_user_photo:
                    # Use the user photo as the base image to edit
                    base_image_png = make_square_rgba_png(user_bytes, size=1024)
                    edit_prompt = (
                        f"The reference strip on the left shows (left to right): the saree pallu pattern, "
                        f"the main fabric, and the blouse. Using EXACTLY these fabrics, colors, and patterns, "
                        f"dress the person shown in this photo in a {drape_style}-style saree drape. "
                        f"Preserve the person's face, skin tone, and body proportions exactly. "
                        f"Show a three-angle composite: front view, 45-degree view, and back view. "
                        f"Professional fashion photography, studio lighting, neutral background.\n\n"
                        f"Additional fabric details from analysis:\n{dalle_prompt}"
                    )
                    # For user-photo mode, send user photo as base + reference composite as second image
                    files = [
                        ("image[]", ("user_photo.png", base_image_png, "image/png")),
                        ("image[]", ("reference.png", reference_composite, "image/png")),
                    ]
                else:
                    # No user photo — send only the reference composite as the base
                    edit_prompt = (
                        f"The image shows a reference strip (left to right): saree pallu pattern, "
                        f"main saree fabric, and blouse. Using EXACTLY these fabrics, colors, and patterns, "
                        f"generate a fashion photo of an elegant Indian woman wearing this saree in "
                        f"{drape_style} drape style. "
                        f"Traditional juda bun with white gajra flowers, minimal natural makeup, "
                        f"traditional gold jewelry. "
                        f"THREE-ANGLE composite: front view (left), 45-degree view (center), back view (right). "
                        f"Premium catalog photography, studio lighting, neutral beige background, full body visible. "
                        f"Accurately reproduce every color, motif, and pattern from the reference images.\n\n"
                        f"Additional fabric details from analysis:\n{dalle_prompt}"
                    )
                    files = [
                        ("image[]", ("reference.png", reference_composite, "image/png")),
                    ]

                img1_response = await client.post(
                    "https://api.openai.com/v1/images/edits",
                    headers={"Authorization": f"Bearer {api_key}"},
                    files=files,
                    data={
                        "model": "gpt-image-1",
                        "prompt": edit_prompt,
                        "n": "1",
                        "size": "1024x1024",
                        "quality": "high",
                    }
                )

                img1_result = img1_response.json()
                print(f"gpt-image-1 response status: {img1_response.status_code}")

                if "error" not in img1_result and img1_result.get("data"):
                    data_item = img1_result["data"][0]
                    if data_item.get("b64_json"):
                        image_b64 = data_item["b64_json"]
                    elif data_item.get("url"):
                        img_download = await client.get(data_item["url"])
                        image_b64 = base64.b64encode(img_download.content).decode("utf-8")
                    model_used = "gpt-image-1"
                    revised_prompt = edit_prompt
                    print("gpt-image-1 succeeded!")
                else:
                    err_msg = img1_result.get("error", {}).get("message", "unknown error")
                    print(f"gpt-image-1 failed: {err_msg}")

            except Exception as e1:
                print(f"gpt-image-1 exception: {traceback.format_exc()}")

            # ---------------------------------------------------------------
            # Step 2b: Fallback to DALL-E 3 (text-only, no image references)
            # ---------------------------------------------------------------
            if not image_b64:
                print("Step 2b: Falling back to DALL-E 3 (text prompt only)...")
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
                        "size": "1792x1024",
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