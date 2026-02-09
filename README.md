# YouTube to Article with Images/GIFs

Given a YouTube URL, this project:
- downloads the transcript,
- generates an article draft (transcript-driven or Gemini-written),
- picks the best visual match for each section (JPG or GIF),
- and exports final Markdown + HTML.

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
Flags:
- `--mock`: skips network calls and uses local mock transcript data.

Real run (recommended default):
```bash
python main.py "https://www.youtube.com/watch?v=VIDEO_ID" --mode transcript --image-profile fast
```
Flags:
- `--mode transcript`: builds article sections directly from transcript timing/text.
- `--image-profile fast`: fastest media matching profile.

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

## Command Recipes (Flags Explained)

Default transcript pipeline (recommended):
```bash
python main.py "https://www.youtube.com/watch?v=VIDEO_ID" --mode transcript --image-profile fast
```
Flags:
- `--mode transcript`: deterministic draft from transcript segmentation.
- `--image-profile fast`: lowest CPU usage and quickest runs.

GIF output with custom length/quality:
```bash
python main.py "https://www.youtube.com/watch?v=VIDEO_ID" --gif --gif-duration 3.2 --gif-fps 10 --gif-width 960
```
Flags:
- `--gif`: output GIF clips instead of JPG frames.
- `--gif-duration 3.2`: each GIF clip lasts 3.2 seconds.
- `--gif-fps 10`: GIF frame rate (higher = smoother, larger files).
- `--gif-width 960`: output width in pixels (higher = sharper, larger files).

Higher-accuracy media matching:
```bash
python main.py "https://www.youtube.com/watch?v=VIDEO_ID" --image-profile accurate --ocr-budget 12
```
Flags:
- `--image-profile accurate`: slower but stronger matching.
- `--ocr-budget 12`: OCR checks for up to 12 sections.

Gemini-written article draft:
```bash
python main.py "https://www.youtube.com/watch?v=VIDEO_ID" --mode gemini
```
Flags:
- `--mode gemini`: Gemini writes `article_draft.md` before media injection.
- Requires `GEMINI_API_KEY` or `GOOGLE_API_KEY`.

Minimal HTML style:
```bash
python main.py "https://www.youtube.com/watch?v=VIDEO_ID" --html-style basic
```
Flags:
- `--html-style basic`: simpler HTML layout.

## Options Cheat Sheet

- `--mode transcript|gemini`: choose how draft text is produced.
- `--image-profile fast|balanced|accurate`: choose speed vs matching quality.
- `--smart-retry`: small OCR retry for weak matches in `fast`/`balanced`.
- `--gif`: switch output media from JPG to GIF.
- `--gif-duration <seconds>`: GIF length per section (default `2.6`).
- `--gif-fps <int>`: GIF frame rate (default `8`).
- `--gif-width <pixels>`: GIF width (default `880`).
- `--frame-offset <seconds>`: how long after section start to begin searching frames.
- `--frame-window <seconds>`: how far ahead to search for good frames.
- `--candidates <int>`: number of candidate frames to score.
- `--html-style article|basic`: choose final HTML style.
- `--mock`: run with local synthetic data for testing.

## Data and copyright

- Files committed in this repository that look like transcript/article data (for example `mock_transcript.json` and `article_draft.md`) are synthetic sample data for testing.
- Do not commit transcripts, screenshots, GIFs, or video files generated from third-party YouTube content unless you have explicit rights to redistribute them.
- Keep generated runs under `output/` (already gitignored) to avoid accidentally publishing copyrighted source material.
