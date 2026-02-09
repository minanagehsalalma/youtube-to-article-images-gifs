# YouTube to Article with Images/GIFs

Convert a YouTube video into a Markdown/HTML article with smart media selection:
- single-frame JPGs (fastest), or
- short GIF clips (better context for moving UI moments).

The media picker uses visual scoring (Pillow), and can use targeted OCR checks (system Tesseract) for weak/ambiguous matches.
HTML export and styling are handled by `scripts/html_renderer.py`.

## Requirements

### System tools
- `ffmpeg` for frame/GIF extraction
- `yt-dlp` for video download
- `tesseract` (optional but recommended for OCR-assisted matching)

### Python packages
```bash
pip install -r requirements.txt
```

Optional (Level B screenshots):
```bash
pip install -U playwright
playwright install chromium
```

## API key

Set `GEMINI_API_KEY` (or `GOOGLE_API_KEY`):

```powershell
setx GEMINI_API_KEY "YOUR_KEY_HERE"
```

Quick validation:
```bash
python test_gemini_key.py
```

## Run examples

Mock mode:
```bash
python main.py "any-url" --mock --level A
```

Real run (default profile is `fast`):
```bash
python main.py "https://www.youtube.com/watch?v=VIDEO_ID" --level A --mode transcript --image-profile fast
```

Enable targeted retry on weak sections:
```bash
python main.py "https://www.youtube.com/watch?v=VIDEO_ID" --level A --image-profile fast --smart-retry
```

Use GIF output instead of JPG frames (default `--gif-duration` is `2.6` seconds):
```bash
python main.py "https://www.youtube.com/watch?v=VIDEO_ID" --level A --gif
```

HTML style presets:
```bash
# Current balanced article style (default)
python main.py "https://www.youtube.com/watch?v=VIDEO_ID" --level A --html-style article

# Minimal/basic style
python main.py "https://www.youtube.com/watch?v=VIDEO_ID" --level A --html-style basic
```

Shortcut flags:
- `--html-article` = `--html-style article`
- `--html-basic` = `--html-style basic`

## Profiles

- `fast`: lowest CPU, visual scoring only
- `balanced`: extra fallback pass before acceptance
- `accurate`: multi-pass plus bounded OCR reranking (`--ocr-budget`)

Outputs are written to `output/<video_id>/` with `article_final.md`, `article_final.html`, and `images/`.

Recommended GitHub repo name: `youtube-to-article-images-gifs`

## Data and copyright

- Files committed in this repository that look like transcript/article data (for example `mock_transcript.json` and `article_draft.md`) are synthetic sample data for testing.
- Do not commit transcripts, screenshots, GIFs, or video files generated from third-party YouTube content unless you have explicit rights to redistribute them.
- Keep generated runs under `output/` (already gitignored) to avoid accidentally publishing copyrighted source material.
