# Repository Guidelines

## Project Structure & Module Organization
`main.py` is the CLI entry point that orchestrates transcript parsing, sectioning, media picking, and article generation. Core modules live in `scripts/` (`article_generator.py`, `downloader.py`, `frame_extractor.py`, `screenshotter.py`, `html_renderer.py`). Generated runtime artifacts are written under `output/` and treated as build output. Sample static data files in the repo include `mock_transcript.json` and `article_draft.md`.

## Build, Test, and Development Commands
- `pip install -r requirements.txt` installs the required Python dependencies.
- `python test_gemini_key.py` validates that `GEMINI_API_KEY` (or `GOOGLE_API_KEY`) is set and usable.
- `python main.py "https://www.youtube.com/watch?v=VIDEO_ID" --mode transcript --image-profile fast` runs a real pipeline execution.
- `python main.py "any-url" --mock` runs in mock mode without network calls.
- `python main.py "https://www.youtube.com/watch?v=VIDEO_ID" --gif` emits GIF clips instead of single-frame JPGs (default GIF duration: `2.6` seconds).
- `python main.py "https://www.youtube.com/watch?v=VIDEO_ID" --html-style basic` renders `article_final.html` with the minimal/basic layout (`--html-style article` is the default balanced layout).

## Coding Style & Naming Conventions
Use Python with 4-space indentation. Keep functions and variables in `snake_case`, and prefer lowercase module names with underscores to match `scripts/` conventions. When adding new code, follow existing patterns for type hints and small, focused helpers. No formatter or linter is configured in the repo, so keep changes PEP 8 friendly and consistent with current style.

## Testing Guidelines
There is no automated test framework configured. Use `python test_gemini_key.py` for API/key validation and `python main.py "any-url" --mock` for offline pipeline smoke tests. If you add tests, keep them lightweight, name them `test_*.py`, and document the run command in this section.

## Commit & Pull Request Guidelines
No Git history is present in this workspace, so commit conventions are not established. Use short, imperative commit messages (for example, "Add mock transcript loader"). For pull requests, include a brief summary, list any commands run (with output notes if relevant), and mention configuration changes such as new environment variables.

## Security & Configuration Tips
Never commit API keys. Use `GEMINI_API_KEY` or `GOOGLE_API_KEY` from the environment, and prefer `--mock` for offline or demo runs. OCR support depends on a system `tesseract` binary; if not installed, the pipeline falls back to visual-only scoring. Keep `output/` out of version control.
