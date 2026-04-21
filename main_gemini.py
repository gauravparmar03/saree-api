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

app = FastAPI(title="Saree AI API (Gemini)")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Helper: encode image bytes to base64 string (for Gemini inline image parts)
# ---------------------------------------------------------------------------
def to_b64(data: bytes) -> str:
    return base64.b64encode(data).decode("utf-8")


def make_square_rgba_png(image_bytes: bytes, size: int = 512) -> bytes:
    """Convert image bytes to a square RGBA PNG at the given size."""
    img = Image.open(io.BytesIO(image_bytes)).convert("RGBA")
    img = img.resize((size, size), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def composite_images_grid(images_bytes: list[bytes], cell_size: int = 512) -> bytes:
    """Arrange images in a horizontal strip, each resized to cell_size x cell_size."""
    n = len(images_bytes)
    composite = Image.new("RGBA", (cell_size * n, cell_size), (255, 255, 255, 255))
    for i, b in enumerate(images_bytes):
        img = Image.open(io.BytesIO(b)).convert("RGBA")
        img = img.resize((cell_size, cell_size), Image.LANCZOS)
        composite.paste(img, (i * cell_size, 0))
    buf = io.BytesIO()
    composite.save(buf, format="PNG")
    return buf.getvalue()


# ---------------------------------------------------------------------------
# Gemini API helpers
# ---------------------------------------------------------------------------
GEMINI_BASE = "https://generativelanguage.googleapis.com/v1beta"
GEMINI_VISION_MODEL   = "gemini-2.0-flash"      # multimodal vision + text
IMAGEN_MODEL          = "imagen-3.0-generate-001" # image generation


def gemini_image_part(image_bytes: bytes, mime_type: str = "image/jpeg") -> dict:
    """Build an inline image part for Gemini's generateContent API."""
    # Gemini accepts JPEG/PNG inline as base64 inlineData
    # Normalise to PNG for reliability
    png_bytes = make_square_rgba_png(image_bytes, size=512)
    return {
        "inlineData": {
            "mimeType": "image/png",
            "data": to_b64(png_bytes)
        }
    }


async def gemini_vision_prompt(
    api_key: str,
    client: httpx.AsyncClient,
    pallu_bytes: bytes,
    blouse_bytes: bytes,
    fabric_bytes: bytes,
    user_bytes: Optional[bytes],
    has_user_photo: bool,
    drape_style: str,
) -> str:
    """
    Call Gemini 2.0 Flash to analyse the reference images and return a
    detailed image-generation prompt for Imagen 3.
    """
    text_prompt = f"""You are an expert fashion analyst and prompt engineer specialising in Indian ethnic wear.

Analyse the provided saree fabric images carefully and extract every detail:

FROM PALLU IMAGE (1st image): Describe exact border design, motifs, colours, patterns, tassel details, zari work.
FROM BLOUSE IMAGE (2nd image): Describe exact colour, neckline shape, sleeve length/style, embroidery, fit.
FROM FABRIC IMAGE (3rd image): Describe exact base colour, weave pattern, motifs, texture, border stripes.

{"FROM USER PHOTO (4th image): Describe the person's skin tone, body proportions, height, facial features to preserve." if has_user_photo else ""}

Then write an image-generation prompt that:

1. FABRIC ACCURACY: Uses exact colours/patterns/motifs you described from the images.
2. DRAPING: {drape_style} saree drape style — wide front pleats, pallu over left shoulder, visible border and tassels, realistic fabric physics.
3. BLOUSE: Exactly matches the blouse image details.
4. {"PERSON: Dress the specific person from the user photo in this saree. Preserve their exact face, skin tone, body proportions. The saree replaces their existing clothing naturally." if has_user_photo else "MODEL: Elegant Indian woman, traditional juda bun with white gajra flowers, minimal natural makeup, traditional gold jewellery (necklace, jhumkas, bangles)."}
5. OUTPUT: THREE-ANGLE composite image — LEFT shows front view full body, CENTRE shows 45-degree side view, RIGHT shows back view showing pallu fall clearly.
6. PHOTOGRAPHY: Premium fashion catalogue shoot, professional studio lighting with soft shadows, neutral beige/cream background, sharp focus throughout, full body visible in all three views.
7. QUALITY: Ultra-realistic, 4K quality, accurate fabric draping physics, no pattern distortion.

Return ONLY the prompt text. Be extremely specific about colours and patterns."""

    parts = [
        {"text": text_prompt},
        gemini_image_part(pallu_bytes),
        gemini_image_part(blouse_bytes),
        gemini_image_part(fabric_bytes),
    ]

    if has_user_photo:
        parts.append(gemini_image_part(user_bytes))

    payload = {
        "contents": [{"role": "user", "parts": parts}],
        "generationConfig": {"maxOutputTokens": 1500, "temperature": 0.4}
    }

    url = f"{GEMINI_BASE}/models/{GEMINI_VISION_MODEL}:generateContent?key={api_key}"
    resp = await client.post(url, json=payload)
    result = resp.json()

    if "error" in result:
        raise ValueError(f"Gemini vision error: {result['error'].get('message')}")

    # Extract text from candidates
    candidates = result.get("candidates", [])
    if not candidates:
        raise ValueError("Gemini vision returned no candidates.")

    parts_out = candidates[0].get("content", {}).get("parts", [])
    text_parts = [p["text"] for p in parts_out if "text" in p]
    return "\n".join(text_parts).strip()


async def imagen3_generate(
    api_key: str,
    client: httpx.AsyncClient,
    prompt: str,
) -> Optional[str]:
    """
    Call Imagen 3 via the Google AI generativelanguage endpoint.
    Returns base64-encoded PNG, or None on failure.

    NOTE: Imagen 3 is text-prompt only — it does NOT accept image references
    directly via this endpoint. Use the detailed GPT/Gemini-analysed prompt.
    """
    url = f"{GEMINI_BASE}/models/{IMAGEN_MODEL}:predict?key={api_key}"
    payload = {
        "instances": [{"prompt": prompt}],
        "parameters": {
            "sampleCount": 1,
            "aspectRatio": "16:9",    # landscape for three-angle composite
            "safetyFilterLevel": "block_few",
            "personGeneration": "allow_adult",
        }
    }

    resp = await client.post(url, json=payload)
    result = resp.json()
    print(f"Imagen 3 response status: {resp.status_code}")

    if "error" in result:
        print(f"Imagen 3 error: {result['error'].get('message')}")
        return None

    predictions = result.get("predictions", [])
    if not predictions:
        print("Imagen 3 returned no predictions.")
        return None

    # Each prediction has a 'bytesBase64Encoded' field
    b64 = predictions[0].get("bytesBase64Encoded")
    if b64:
        print("Imagen 3 succeeded!")
        return b64

    print("Imagen 3 prediction had no image data.")
    return None


async def gemini_native_image_generate(
    api_key: str,
    client: httpx.AsyncClient,
    prompt: str,
    reference_images_bytes: list[bytes],
) -> Optional[str]:
    """
    Fallback: use Gemini 2.0 Flash's native image OUTPUT capability
    (responseModalities: IMAGE) with reference images as visual context.
    Returns base64 PNG or None.

    This is available on gemini-2.0-flash-preview-image-generation.
    """
    model = "gemini-2.0-flash-preview-image-generation"
    url = f"{GEMINI_BASE}/models/{model}:generateContent?key={api_key}"

    parts = []
    for img_bytes in reference_images_bytes:
        parts.append(gemini_image_part(img_bytes))

    parts.append({
        "text": (
            "Using the reference images above as fabric/design references, "
            "generate a photorealistic fashion image:\n\n" + prompt
        )
    })

    payload = {
        "contents": [{"role": "user", "parts": parts}],
        "generationConfig": {
            "responseModalities": ["IMAGE", "TEXT"],
            "temperature": 1.0,
        }
    }

    resp = await client.post(url, json=payload)
    result = resp.json()
    print(f"Gemini native image gen status: {resp.status_code}")

    if "error" in result:
        print(f"Gemini image gen error: {result['error'].get('message')}")
        return None

    candidates = result.get("candidates", [])
    if not candidates:
        return None

    for part in candidates[0].get("content", {}).get("parts", []):
        inline = part.get("inlineData", {})
        if inline.get("data"):
            print("Gemini native image generation succeeded!")
            return inline["data"]   # already base64

    return None


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/")
def root():
    api_key = os.environ.get("GEMINI_API_KEY", "")
    return {
        "status": "Saree AI API (Gemini) is running",
        "gemini_key_set": bool(api_key),
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
        api_key = os.environ.get("GEMINI_API_KEY", "")
        if not api_key:
            return JSONResponse(
                status_code=500,
                content={"error": "GEMINI_API_KEY is not set in environment variables."}
            )

        # Read all images
        pallu_bytes  = await pallu_image.read()
        blouse_bytes = await blouse_image.read()
        fabric_bytes = await fabric_image.read()

        has_user_photo = False
        user_bytes = None
        if user_photo and user_photo.filename:
            user_bytes = await user_photo.read()
            if user_bytes:
                has_user_photo = True

        async with httpx.AsyncClient(timeout=240.0) as client:

            # -----------------------------------------------------------
            # Step 1: Gemini 2.0 Flash Vision → detailed Imagen 3 prompt
            # -----------------------------------------------------------
            print("Step 1: Analysing images with Gemini 2.0 Flash Vision...")
            dalle_prompt = await gemini_vision_prompt(
                api_key, client,
                pallu_bytes, blouse_bytes, fabric_bytes,
                user_bytes, has_user_photo,
                drape_style,
            )
            print(f"Generated prompt ({len(dalle_prompt)} chars): {dalle_prompt[:300]}...")

            # -----------------------------------------------------------
            # Step 2a: Imagen 3 (text-to-image, best quality)
            # -----------------------------------------------------------
            image_b64  = None
            model_used = None
            revised_prompt = dalle_prompt

            image_b64 = await imagen3_generate(api_key, client, dalle_prompt)
            if image_b64:
                model_used = "imagen-3"

            # -----------------------------------------------------------
            # Step 2b: Fallback — Gemini 2.0 Flash native image output
            #          (accepts reference images directly as visual context)
            # -----------------------------------------------------------
            if not image_b64:
                print("Step 2b: Falling back to Gemini native image generation...")
                ref_images = [pallu_bytes, fabric_bytes, blouse_bytes]
                if has_user_photo:
                    ref_images.append(user_bytes)

                image_b64 = await gemini_native_image_generate(
                    api_key, client, dalle_prompt, ref_images
                )
                if image_b64:
                    model_used = "gemini-2.0-flash-image-gen"

            if not image_b64:
                return JSONResponse(
                    status_code=500,
                    content={
                        "error": "Both Imagen 3 and Gemini image generation failed.",
                        "dalle_prompt": dalle_prompt,
                    }
                )

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
        return JSONResponse(
            status_code=500,
            content={"error": str(e), "traceback": detail}
        )