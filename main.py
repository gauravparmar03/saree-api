import os
import base64
import traceback
import httpx
import io
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from typing import Optional
from PIL import Image, ImageFilter

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


def resize_to_square_png(image_bytes: bytes, size: int = 1024) -> bytes:
    """Resize image to exact square with white padding, return PNG bytes."""
    img = Image.open(io.BytesIO(image_bytes)).convert("RGBA")
    img.thumbnail((size, size), Image.LANCZOS)
    canvas = Image.new("RGBA", (size, size), (255, 255, 255, 255))
    offset = ((size - img.width) // 2, (size - img.height) // 2)
    canvas.paste(img, offset, img)
    buf = io.BytesIO()
    canvas.save(buf, format="PNG")
    return buf.getvalue()


def composite_images_grid(images_bytes: list[bytes], cell_size: int = 512) -> bytes:
    """Arrange images in a horizontal strip. Returns PNG bytes."""
    n = len(images_bytes)
    composite = Image.new("RGBA", (cell_size * n, cell_size), (255, 255, 255, 255))
    for i, b in enumerate(images_bytes):
        img = Image.open(io.BytesIO(b)).convert("RGBA")
        img.thumbnail((cell_size, cell_size), Image.LANCZOS)
        x = i * cell_size + (cell_size - img.width) // 2
        y = (cell_size - img.height) // 2
        composite.paste(img, (x, y), img)
    buf = io.BytesIO()
    composite.save(buf, format="PNG")
    return buf.getvalue()


def generate_clothing_mask(image_bytes: bytes, size: int = 1024) -> bytes:
    """
    Generate a mask PNG for gpt-image-1 /images/edits:
      - TRANSPARENT pixels = areas to replace (clothing/body)
      - OPAQUE WHITE pixels = areas to keep (face, background)

    Uses rembg to isolate the person, then preserves the top 28% of
    the person's bounding box (head/neck/shoulders) and makes the
    rest transparent so the saree gets painted there.

    Falls back to a simple top-25% cutoff if rembg is unavailable.
    """
    try:
        from rembg import remove

        img = Image.open(io.BytesIO(image_bytes)).convert("RGBA")
        img.thumbnail((size, size), Image.LANCZOS)
        canvas = Image.new("RGBA", (size, size), (255, 255, 255, 255))
        offset = ((size - img.width) // 2, (size - img.height) // 2)
        canvas.paste(img, offset, img)

        # Isolate person — alpha channel tells us where the person is
        no_bg = remove(canvas)
        alpha = no_bg.split()[3]

        bbox = alpha.getbbox()
        mask = Image.new("RGBA", (size, size), (255, 255, 255, 255))  # all white = keep

        if bbox:
            left, upper, right, lower = bbox
            face_cutoff_row = upper + int((lower - upper) * 0.28)  # keep head + shoulders

            alpha_data = list(alpha.getdata())
            mask_data = list(mask.getdata())

            for idx, a in enumerate(alpha_data):
                row = idx // size
                # Pixel is part of the person's body (below face) → make transparent = replace
                if a > 30 and row > face_cutoff_row:
                    mask_data[idx] = (0, 0, 0, 0)

            mask.putdata(mask_data)

        # Slightly blur mask edges to avoid hard seams
        mask = mask.filter(ImageFilter.GaussianBlur(radius=3))

        buf = io.BytesIO()
        mask.save(buf, format="PNG")
        return buf.getvalue()

    except ImportError:
        print("rembg not available — using simple geometric mask fallback")
        mask = Image.new("RGBA", (size, size), (255, 255, 255, 255))
        mask_data = list(mask.getdata())
        face_cutoff = int(size * 0.25)
        for idx in range(len(mask_data)):
            if idx // size > face_cutoff:
                mask_data[idx] = (0, 0, 0, 0)
        mask.putdata(mask_data)
        buf = io.BytesIO()
        mask.save(buf, format="PNG")
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
        # Step 1: GPT-4o Vision — analyze all images, return a rich prompt
        # ---------------------------------------------------------------
        vision_content = [
            {
                "type": "text",
                "text": f"""You are an expert fashion analyst and prompt engineer specializing in Indian ethnic wear.

Analyze the provided saree fabric images VERY carefully:

FROM PALLU IMAGE (image 1): Describe exact border design, motifs, colors, patterns, tassel/zari details
FROM BLOUSE IMAGE (image 2): Describe exact color, neckline shape, sleeve length/style, embroidery
FROM FABRIC IMAGE (image 3): Describe exact base color, weave pattern, motifs, texture, border stripes
{"FROM USER PHOTO (image 4): Describe the person's skin tone, hair color, facial features, body proportions — these MUST be preserved." if has_user_photo else ""}

Then write an image editing instruction that:
1. FABRIC: Uses exact colors/patterns/motifs from the images — be very specific
2. DRAPING: {drape_style} style — wide front pleats, pallu over left shoulder, visible border/tassels, realistic fabric physics
3. BLOUSE: Exactly matches image 2 — same color, neckline, sleeves, embroidery
4. {"PERSON: Preserve the person's face, skin tone, hair, and body proportions exactly. Only replace the clothing." if has_user_photo else "MODEL: Elegant Indian woman, juda bun with white gajra, minimal makeup, gold jewelry (necklace, jhumkas, bangles)"}
5. PHOTOGRAPHY: Professional catalog shoot, soft studio lighting, neutral beige background, full body in frame
6. QUALITY: Ultra-realistic, 4K, accurate draping physics

Return ONLY the prompt text."""
            },
            {"type": "image_url", "image_url": {"url": f"data:{pallu_image.content_type or 'image/jpeg'};base64,{base64.b64encode(pallu_bytes).decode()}", "detail": "high"}},
            {"type": "image_url", "image_url": {"url": f"data:{blouse_image.content_type or 'image/jpeg'};base64,{base64.b64encode(blouse_bytes).decode()}", "detail": "high"}},
            {"type": "image_url", "image_url": {"url": f"data:{fabric_image.content_type or 'image/jpeg'};base64,{base64.b64encode(fabric_bytes).decode()}", "detail": "high"}},
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
            vision_resp = await client.post(
                "https://api.openai.com/v1/chat/completions",
                headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
                json={
                    "model": "gpt-4o",
                    "messages": [{"role": "user", "content": vision_content}],
                    "max_tokens": 1500,
                }
            )
            vision_result = vision_resp.json()
            if "error" in vision_result:
                return JSONResponse(status_code=500, content={"error": f"GPT-4o error: {vision_result['error'].get('message')}"})

            dalle_prompt = vision_result["choices"][0]["message"]["content"].strip()
            print(f"Prompt ({len(dalle_prompt)} chars): {dalle_prompt[:300]}...")

            # ---------------------------------------------------------------
            # Step 2a: gpt-image-1 /images/edits with real image references
            # ---------------------------------------------------------------
            image_b64 = None
            model_used = None
            revised_prompt = dalle_prompt

            try:
                print("Step 2a: gpt-image-1 /images/edits...")

                fabric_composite = composite_images_grid(
                    [pallu_bytes, fabric_bytes, blouse_bytes], cell_size=512
                )

                if has_user_photo:
                    # --------------------------------------------------------
                    # USER PHOTO FLOW
                    # - base image  : user photo (1024×1024 square PNG)
                    # - mask        : transparent over clothing area, white over face/bg
                    # - image[]     : fabric composite so model sees what saree looks like
                    # The model fills transparent regions with the saree while keeping face intact
                    # --------------------------------------------------------
                    print("  Generating clothing mask...")
                    base_png = resize_to_square_png(user_bytes, size=1024)
                    mask_png = generate_clothing_mask(user_bytes, size=1024)

                    edit_prompt = (
                        f"Drape a {drape_style}-style saree on the person in this photo. "
                        f"Fill the transparent clothing area with the saree. "
                        f"Use EXACTLY the fabrics and colors from the reference strip: "
                        f"left panel = pallu/border pattern, center panel = main fabric, right panel = blouse. "
                        f"Keep the person's face, skin tone, hair, and body proportions 100% identical — "
                        f"do not change anything above the shoulders. "
                        f"Saree: wide front pleats, pallu draped over left shoulder, visible decorative border and tassels. "
                        f"Blouse matches the right panel of the reference exactly. "
                        f"Professional studio lighting, neutral background.\n\n"
                        f"Detailed fabric description:\n{dalle_prompt}"
                    )

                    files = [
                        ("image",   ("user_photo.png", base_png,          "image/png")),
                        ("mask",    ("mask.png",        mask_png,          "image/png")),
                        ("image[]", ("fabrics.png",     fabric_composite,  "image/png")),
                    ]

                else:
                    # --------------------------------------------------------
                    # NO USER PHOTO FLOW
                    # Pass fabric composite as the base; model generates a model wearing it
                    # --------------------------------------------------------
                    edit_prompt = (
                        f"The reference image shows (left to right): saree pallu/border pattern, "
                        f"main saree fabric, and blouse. "
                        f"Generate a full-body fashion photo of an elegant Indian woman wearing this saree "
                        f"draped in {drape_style} style. "
                        f"Reproduce EXACTLY the colors, patterns, and motifs from the reference. "
                        f"Traditional juda bun with white gajra flowers, minimal makeup, gold jewelry. "
                        f"THREE-ANGLE composite: front view (left), 45-degree view (center), back view (right). "
                        f"Premium catalog photography, studio lighting, neutral beige background, full body visible.\n\n"
                        f"Detailed fabric description:\n{dalle_prompt}"
                    )

                    files = [
                        ("image", ("fabrics.png", fabric_composite, "image/png")),
                    ]

                img1_resp = await client.post(
                    "https://api.openai.com/v1/images/edits",
                    headers={"Authorization": f"Bearer {api_key}"},
                    files=files,
                    data={"model": "gpt-image-1", "prompt": edit_prompt, "n": "1", "size": "1024x1024", "quality": "high"},
                )

                img1_result = img1_resp.json()
                print(f"gpt-image-1 HTTP {img1_resp.status_code}")

                if "error" not in img1_result and img1_result.get("data"):
                    item = img1_result["data"][0]
                    if item.get("b64_json"):
                        image_b64 = item["b64_json"]
                    elif item.get("url"):
                        dl = await client.get(item["url"])
                        image_b64 = base64.b64encode(dl.content).decode("utf-8")
                    model_used = "gpt-image-1"
                    revised_prompt = edit_prompt
                    print("gpt-image-1 succeeded!")
                else:
                    print(f"gpt-image-1 failed: {img1_result.get('error', {}).get('message', 'unknown')}")

            except Exception:
                print(f"gpt-image-1 exception:\n{traceback.format_exc()}")

            # ---------------------------------------------------------------
            # Step 2b: Fallback — DALL-E 3 (text only)
            # ---------------------------------------------------------------
            if not image_b64:
                print("Step 2b: Falling back to DALL-E 3...")
                dalle_resp = await client.post(
                    "https://api.openai.com/v1/images/generations",
                    headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
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
                dalle_result = dalle_resp.json()
                if "error" in dalle_result:
                    return JSONResponse(
                        status_code=500,
                        content={"error": f"DALL-E 3 error: {dalle_result['error'].get('message')}", "dalle_prompt": dalle_prompt}
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
            "dalle_prompt": dalle_prompt,
        })

    except Exception as e:
        detail = traceback.format_exc()
        print("ERROR:\n", detail)
        return JSONResponse(status_code=500, content={"error": str(e), "traceback": detail})