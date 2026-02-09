import os
import sys
import json


def _build_prompt(transcript: list[dict]) -> str:
    # Include timestamps to help place images.
    full_text = ""
    for entry in transcript:
        start = float(entry.get("start", 0))
        text = str(entry.get("text", "")).strip()
        full_text += f"[{start:.2f}] {text}\n"

    # Keep prompt size reasonable.
    full_text = full_text[:15000]

    return f"""
You are a professional tech blogger. I will provide you with a transcript of a YouTube video about Chrome extensions.
Your task is to write a high-quality, SEO-optimized blog article based on this transcript.

Requirements:
1. Structure: Use clear headings (H1, H2, H3).
2. Content: Summarize the main points, list each extension mentioned, describe what it does, and why it's useful.
3. Image placeholders: Based on the timestamps in the transcript, suggest where to insert screenshots.
   Use the format: [IMAGE_PLACEHOLDER: description of what should be in the image at timestamp XX.XX]
4. Tone: Professional yet engaging.
5. Format: Output in Markdown.

Transcript:
{full_text}
"""


def generate_article(transcript_json_path: str, output_path: str, model: str | None = None) -> bool:
    """Generate an article draft from a transcript using the Gemini API.

    Expects GEMINI_API_KEY (or GOOGLE_API_KEY) to be set in the environment.
    """
    model = model or os.getenv("GEMINI_MODEL", "gemini-2.5-flash")

    with open(transcript_json_path, "r", encoding="utf-8") as f:
        transcript = json.load(f)

    prompt = _build_prompt(transcript)

    # Lazy import so mock mode can run without google-genai installed.
    try:
        from google import genai
        from google.genai import types
    except Exception as e:
        print(
            "google-genai is not installed. Install with:\n"
            "  pip install -U google-genai\n"
            f"Details: {e}"
        )
        return False

    # Client reads API key from GEMINI_API_KEY / GOOGLE_API_KEY env vars.
    client = genai.Client()

    try:
        resp = client.models.generate_content(
            model=model,
            contents=prompt,
            config=types.GenerateContentConfig(
                system_instruction=(
                    "You are a helpful assistant that writes blog articles from video transcripts. "
                    "Follow the user's formatting requirements exactly."
                ),
                temperature=0.4,
                max_output_tokens=4096,
            ),
        )
        article_content = (resp.text or "").strip()

        with open(output_path, "w", encoding="utf-8") as f:
            f.write(article_content)
        return True
    except Exception as e:
        print(f"Error generating article: {e}")
        return False
    finally:
        try:
            client.close()
        except Exception:
            pass


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python3 article_generator.py <transcript_json_path> <output_path> [model]")
        sys.exit(1)

    t_path = sys.argv[1]
    o_path = sys.argv[2]
    m = sys.argv[3] if len(sys.argv) > 3 else None

    if generate_article(t_path, o_path, m):
        print(f"Article generated and saved to {o_path}")
