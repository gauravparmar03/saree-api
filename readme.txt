1. User uploads images
   - Pallu image
   - Blouse image
   - Fabric image
   - (Optional) User photo

2. Images are read as bytes

3. GPT-4o Vision analyzes images
   - Extracts colors, patterns, textures
   - Understands blouse design
   - (Optional) Detects user appearance

4. AI generates a detailed prompt
   - Ensures fabric accuracy
   - Defines draping style
   - Adds photography instructions

5. Reference image strip is created
   - Combines pallu + fabric + blouse into one image

6. Image generation (Primary)
   - Uses `gpt-image-1` with real image references

7. Fallback (if primary fails)
   - Uses DALL·E 3 (text-only prompt)

8. Final image is returned
   - Base64 encoded image
   - Model used
   - Prompt details