# YouTube to Article with Images/GIFs

Turn a YouTube video into a Markdown/HTML article with auto-selected media:
- JPG frames (fastest), or
- GIF clips (better for moving UI moments).

## What You Get

Each run writes to `output/<video_id>/`:
- `transcript.json`
- `article_draft.md`
- `article_final.md`
- `article_final.html`
- `images/`

## Install

### 1) Python packages
```bash
pip install -r requirements.txt
```

### 2) System tools
- `ffmpeg` (required for frames/GIFs)
- `yt-dlp` (required for video download)
- `tesseract` (optional, improves OCR matching in accurate mode)

## Quick Start

Mock/offline smoke test:
```bash
python main.py "any-url" --mock
```

Real run (recommended default):
```bash
python main.py "https://www.youtube.com/watch?v=VIDEO_ID" --mode transcript --image-profile fast
```

## Modes (Important)

- `--mode transcript` (default):
  - No Gemini API key needed.
  - Builds article sections directly from transcript timing/content.
  - More deterministic and easier to debug.

- `--mode gemini`:
  - Requires `GEMINI_API_KEY` (or `GOOGLE_API_KEY`).
  - Gemini writes the draft article text.
  - Can be more natural, but less deterministic.

Gemini mode example:
```bash
python main.py "https://www.youtube.com/watch?v=VIDEO_ID" --mode gemini
```

## Gemini API Key (Only if Using `--mode gemini`)

```powershell
setx GEMINI_API_KEY "YOUR_KEY_HERE"
```

Validate key:
```bash
python test_gemini_key.py
```

## Most Useful Options

- `--gif`: output GIF clips instead of JPG frames.
- `--image-profile fast|balanced|accurate`:
  - `fast`: fastest, lowest CPU.
  - `balanced`: better matching with modest extra work.
  - `accurate`: best matching, slowest, can use OCR (`--ocr-budget`).
- `--smart-retry`: lightweight OCR retry for weak matches in `fast`/`balanced`.
- `--html-style article|basic`: final HTML style.
- `--mock`: run without network calls for testing.

Examples:
```bash
# GIF output
python main.py "https://www.youtube.com/watch?v=VIDEO_ID" --gif

# Better matching profile
python main.py "https://www.youtube.com/watch?v=VIDEO_ID" --image-profile balanced

# Gemini-written draft
python main.py "https://www.youtube.com/watch?v=VIDEO_ID" --mode gemini
```

## Data and copyright

- Files committed in this repository that look like transcript/article data (for example `mock_transcript.json` and `article_draft.md`) are synthetic sample data for testing.
- Do not commit transcripts, screenshots, GIFs, or video files generated from third-party YouTube content unless you have explicit rights to redistribute them.
- Keep generated runs under `output/` (already gitignored) to avoid accidentally publishing copyrighted source material.
