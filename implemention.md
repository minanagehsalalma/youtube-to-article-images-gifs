# Implementation Notes

This document explains how the pipeline works end-to-end so future research and iteration can happen quickly and safely.

## 1. Goal and Runtime Contract

Input:
- A YouTube URL (or mock mode), plus CLI tuning flags.

Output (under `output/<video_id>/`):
- `transcript.json`
- `article_draft.md`
- `article_final.md`
- `article_final.html`
- `images/` with either `.jpg` frames or `.gif` clips

Core entrypoint:
- `main.py`

Core support modules:
- `scripts/downloader.py`
- `scripts/article_generator.py`
- `scripts/html_renderer.py`
- `scripts/frame_extractor.py` (utility, not in the main runtime path)

## 2. High-Level Pipeline

`run_pipeline(...)` in `main.py` orchestrates the flow:

1. Resolve `video_id` and create work directories.
2. Fetch transcript:
   - Real: `youtube-transcript-api`
   - Mock: copy `mock_transcript.json`
3. Build article draft:
   - `--mode transcript`: deterministic transcript segmentation + placeholders
   - `--mode gemini`: call `scripts/article_generator.generate_article(...)`
4. Media extraction and rendering:
   - Download video with `yt-dlp` (or reuse existing `video.mp4`)
   - Replace placeholders by selecting best media near section timestamps
   - Export final Markdown and HTML

## 3. Transcript Segmentation Strategy

### 3.1 Numbered marker detection

Functions:
- `find_numbered_items`
- `_looks_like_numbered_list`
- `_fill_small_gaps`

Behavior:
- Uses regex markers like `Number X`, `Tip X`, `Step X`, etc.
- Converts both numeric and word forms via `parse_num`.
- If the marker sequence density looks like a true list, it uses this strategy.
- Fills small numbering gaps (default max gap = 2) by interpolation.

### 3.2 Time-chunk fallback

Functions:
- `build_time_sections`
- `_infer_title_from_text`

Behavior:
- Used when numbered markers are weak/inconsistent.
- Splits transcript into timed sections.
- Estimates title from sentence content / token frequency.

### 3.3 Section payload

Function:
- `attach_section_text`

Behavior:
- Maps each section to the transcript interval until next section start.
- Stores `raw_text` for summary and placeholder generation.

## 4. Draft Generation Paths

### 4.1 Transcript mode

Function:
- `build_transcript_article`

Produces deterministic markdown:
- `## n. Title`
- short section summary from transcript text
- `[IMAGE_PLACEHOLDER: <label> at timestamp MM:SS]`

### 4.2 Gemini mode

Module:
- `scripts/article_generator.py`

Behavior:
- Builds a transcript prompt with timestamps.
- Calls Gemini via `google-genai`.
- Writes model output directly to `article_draft.md`.

## 5. Media Selection Engine

Main function:
- `inject_best_images`

### 5.1 Placeholder parsing

Regex:
- `PLACEHOLDER_LINE_RE`

For each placeholder:
- computes section start (`t0`) and optional next section boundary (`next_t`)
- derives a bounded search interval

### 5.2 Candidate extraction and scoring

Function:
- `extract_best_frame`

Process:
1. Sample multiple candidate timestamps in the search window.
2. Extract frame(s) with ffmpeg.
3. Score each candidate visually (`_score_image`).
4. Optional stability bonus (`_stability_score`) from nearby frame delta.
5. Optional OCR rerank on top-N candidates (`ocr_top_k` only where enabled).

### 5.3 Visual score heuristic (`_score_image`)

Uses Pillow features:
- edge variance and edge intensity
- strong edge ratio
- grayscale variance
- brightness/darkness ratios
- saturation penalty
- center-region skin-like penalty (to demote hand/face overlays)

If Pillow unavailable:
- falls back to file-size proxy (`_file_size_score`).

### 5.4 OCR and semantic retry

Functions:
- `_resolve_tesseract_cmd`
- `_ocr_text`
- `_label_keywords`
- `_ocr_keyword_score`

Design principle:
- OCR is targeted, not global.
- It validates selected weak/ambiguous sections only.
- This avoids expensive OCR on every candidate/frame.

Flow:
1. Build keywords from label.
2. OCR selected frame and score keyword hits/fuzzy matches.
3. If no hits and budget remains, run a small OCR-assisted reselection pass.

### 5.5 Profile behaviors

`--image-profile fast`:
- lowest CPU
- no early/mid extra passes
- no default OCR budget

`--image-profile balanced`:
- stability enabled
- early fallback pass enabled

`--image-profile accurate`:
- stability + early + mid pass
- default OCR budget enabled

`--smart-retry`:
- enables a small targeted OCR budget for `fast`/`balanced`.

## 6. Frame vs GIF Output

Flag:
- `--gif`

Frame mode:
- emits one JPG per section.

GIF mode:
- emits short clip around chosen timestamp via `_ffmpeg_gif`.
- clip is bounded by next section start to reduce overlap.
- tunables:
  - `--gif-duration`
  - `--gif-fps`
  - `--gif-width`

## 7. HTML Rendering

Module:
- `scripts/html_renderer.py`

API:
- `md_to_html_basic(md_text, style="article")`

Styles:
- `basic`: minimal/default-like rendering
- `article`: balanced editorial style

CLI flags:
- `--html-style {basic,article}`
- `--html-basic` shortcut
- `--html-article` shortcut

`main.py` keeps a compatibility wrapper:
- `md_to_html_basic(...)` delegates to `scripts.html_renderer`.

## 8. External Dependencies and Lazy Import Strategy

Python packages:
- `google-genai`
- `youtube-transcript-api`
- `pillow`

System tools:
- `yt-dlp`
- `ffmpeg`
- `tesseract` (optional but recommended)

Lazy imports are used for optional paths, so mock/basic flows fail less often when optional libs are missing.

## 9. Known Boundaries and Research Opportunities

Current boundaries:
- Heading/title inference is heuristic and English-centric.
- OCR currently uses Tesseract with fixed settings (`--oem 1 --psm 6 -l eng`).

Research opportunities:
- Add language-aware OCR/transcript handling.
- Learn score weights from labeled datasets instead of fixed heuristics.
- Add motion-aware GIF quality metrics and scene-change awareness.
- Integrate confidence diagnostics into final markdown/html output.
- Add benchmark suite for precision/recall of frame-label alignment.

## 10. Practical Repro Commands

Real run, fast profile, article HTML:
```bash
python main.py "https://www.youtube.com/watch?v=VIDEO_ID" --mode transcript --image-profile fast --html-style article
```

Real run, GIF output:
```bash
python main.py "https://www.youtube.com/watch?v=VIDEO_ID" --mode transcript --gif
```

Basic HTML style:
```bash
python main.py "https://www.youtube.com/watch?v=VIDEO_ID" --mode transcript --html-basic
```

Mock smoke test:
```bash
python main.py "any-url" --mock
```

Unit tests:
```bash
python -m unittest -v test_main.py
```
