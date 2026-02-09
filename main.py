import os
import sys
import argparse
import asyncio
import shutil
import json
import re
import subprocess
from difflib import SequenceMatcher
from collections import Counter
from pathlib import Path

from scripts.html_renderer import md_to_html_basic as _md_to_html_basic_impl

# ----------------------------
# Helpers
# ----------------------------

def _project_root() -> str:
    return os.path.dirname(os.path.abspath(__file__))


WORD_NUMS = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
    "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14, "fifteen": 15,
    "sixteen": 16, "seventeen": 17, "eighteen": 18, "nineteen": 19, "twenty": 20,
    "thirty": 30, "forty": 40, "fifty": 50, "sixty": 60, "seventy": 70, "eighty": 80, "ninety": 90
}

NUM_WORD = r"(?:\d{1,3}|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|twenty|thirty|forty|fifty|sixty|seventy|eighty|ninety)"
LIST_MARKER_RE = re.compile(
    rf"\b(?:number|no\.?|item|tip|step|point|part|chapter|tool|extension|lesson)\s+(?P<num>{NUM_WORD})\b",
    re.IGNORECASE,
)

STOP_WORDS = {
    "a", "an", "and", "are", "as", "at", "be", "been", "but", "by", "for", "from", "get", "has", "have",
    "if", "in", "into", "is", "it", "its", "just", "like", "more", "not", "now", "of", "on", "or", "our",
    "out", "so", "than", "that", "the", "their", "them", "then", "there", "these", "they", "this", "to",
    "up", "use", "using", "very", "was", "we", "what", "when", "which", "with", "you", "your"
}

def parse_num(token: str) -> int | None:
    token = token.strip().lower().replace("-", " ")
    if token.isdigit():
        return int(token)
    if token in WORD_NUMS:
        return WORD_NUMS[token]
    parts = [p for p in token.split() if p and p != "and"]
    if not parts:
        return None
    total = 0
    for p in parts:
        val = WORD_NUMS.get(p)
        if val is None:
            return None
        total += val
    return total if total > 0 else None

def mmss(seconds: float) -> str:
    m = int(seconds // 60)
    s = int(seconds % 60)
    return f"{m:02d}:{s:02d}"

def safe_slug(s: str) -> str:
    s = re.sub(r"[^\w\-]+", "_", s.lower()).strip("_")
    s = re.sub(r"_+", "_", s)
    return (s[:60] or "item")

def short_summary(text: str, max_sentences: int = 2) -> str:
    parts = re.split(r"(?<=[.!?])\s+", (text or "").strip())
    parts = [p.strip() for p in parts if p.strip()]
    return " ".join(parts[:max_sentences])

def trim_section_lead_in(text: str, section_n: int) -> str:
    s = re.sub(r"\s+", " ", (text or "").strip())
    if not s:
        return s
    marker = re.search(
        rf"\b(?:number|no\.?|item|tip|step|part|point|chapter|tool|extension|lesson)\s+{section_n}\b",
        s,
        re.IGNORECASE,
    )
    if not marker:
        marker = re.search(
            rf"\b(?:number|no\.?|item|tip|step|part|point|chapter|tool|extension|lesson)\s+{NUM_WORD}\b",
            s,
            re.IGNORECASE,
        )
    if marker and marker.start() <= 120:
        s = s[marker.start():]
    return s

def load_transcript(transcript_path: str) -> list[dict]:
    data = json.loads(Path(transcript_path).read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError("Transcript JSON is not a list")
    return data

def _segment_index_for_offset(transcript: list[dict], i: int, window_size: int, char_offset: int) -> int:
    acc = 0
    for j in range(i, min(len(transcript), i + window_size)):
        seg_text = str(transcript[j].get("text", ""))
        seg_len = len(seg_text) + 1
        if acc <= char_offset < acc + seg_len:
            return j
        acc += seg_len
    return i

def _clean_heading_text(text: str, fallback: str) -> str:
    s = re.sub(r"\s+", " ", (text or "").strip(" .,:;-"))
    if not s:
        return fallback
    s = re.sub(r"^(?:and|so|now|okay|alright|all right|well)\s+", "", s, flags=re.IGNORECASE)
    words = re.findall(r"[A-Za-z0-9']+", s)
    if len(words) < 1:
        return fallback
    if len(words) == 1 and len(words[0]) < 3:
        return fallback
    title = " ".join(words[:10])
    return title.lower().capitalize()

def _extract_name_after_marker(window: str, match: re.Match, fallback: str) -> str:
    tail = window[match.end():]
    tail = re.sub(r"^[\s,:\-.]+", "", tail)
    stop = re.search(rf"(?:[.!?]|\b(?:number|tip|step|part|item|chapter|point|tool)\s+{NUM_WORD}\b)", tail, re.IGNORECASE)
    if stop:
        tail = tail[:stop.start()]
    return _clean_heading_text(tail, fallback)

def find_numbered_items(transcript: list[dict], max_items: int = 50) -> list[dict]:
    """
    Detect list-like markers (Number X, Tip X, Step X, etc.) from rolling transcript windows.
    Returns: [{n, name, start, seg_i}]
    """
    detected = []
    window_size = 8

    for i in range(len(transcript)):
        window = " ".join(str(seg.get("text", "")) for seg in transcript[i:i + window_size])
        if not window:
            continue

        match = LIST_MARKER_RE.search(window)
        if not match:
            continue
        if match.start() > 120:
            # Marker is likely for a later segment and will be caught in a later window.
            continue

        n = parse_num(match.group("num"))
        if not n or not (1 <= n <= max_items):
            continue

        seg_i = _segment_index_for_offset(transcript, i, window_size, match.start())
        start = float(transcript[seg_i].get("start", transcript[i].get("start", 0.0)))
        name = _extract_name_after_marker(window, match, fallback=f"Item {n}")
        detected.append({"n": n, "name": name, "start": start, "seg_i": seg_i})

    # Deduplicate by number, keep earliest marker.
    best: dict[int, dict] = {}
    for it in detected:
        n = int(it["n"])
        if n not in best or float(it["start"]) < float(best[n]["start"]):
            best[n] = it

    items = [best[n] for n in sorted(best.keys())]
    return items

def _looks_like_numbered_list(items: list[dict], max_items: int) -> bool:
    if len(items) < 4:
        return False
    nums = sorted({int(it["n"]) for it in items})
    if nums[0] > 3:
        return False
    span = (nums[-1] - nums[0]) + 1
    density = len(nums) / max(1, span)
    return density >= 0.55 or len(nums) >= max(8, min(15, max_items // 2))

def _fill_small_gaps(items: list[dict], max_items: int, max_gap: int = 2) -> list[dict]:
    if len(items) < 2:
        return items
    ordered = sorted(items, key=lambda x: int(x["n"]))
    out = [ordered[0]]
    for prev, curr in zip(ordered, ordered[1:]):
        prev_n = int(prev["n"])
        curr_n = int(curr["n"])
        gap = curr_n - prev_n - 1
        if 0 < gap <= max_gap:
            prev_start = float(prev["start"])
            curr_start = float(curr["start"])
            step = (curr_start - prev_start) / (gap + 1) if curr_start > prev_start else 10.0
            for k in range(1, gap + 1):
                n = prev_n + k
                if n > max_items:
                    continue
                out.append({
                    "n": n,
                    "name": f"Interpolated item {n}",
                    "start": prev_start + (step * k),
                    "seg_i": -1
                })
        out.append(curr)
    return [it for it in out if 1 <= int(it["n"]) <= max_items]

def _infer_title_from_text(text: str, index: int) -> str:
    clean = re.sub(r"\s+", " ", (text or "").strip())
    if not clean:
        return f"Section {index}"

    for sentence in re.split(r"(?<=[.!?])\s+", clean):
        sentence = sentence.strip()
        words = re.findall(r"[A-Za-z0-9']+", sentence)
        if len(words) >= 4:
            return _clean_heading_text(sentence, fallback=f"Section {index}")

    words = [w.lower() for w in re.findall(r"[A-Za-z][A-Za-z'-]{2,}", clean)]
    words = [w for w in words if w not in STOP_WORDS]
    if words:
        top = [w for w, _ in Counter(words).most_common(4)]
        return _clean_heading_text(" ".join(top), fallback=f"Section {index}")
    return f"Section {index}"

def build_time_sections(transcript: list[dict], max_items: int = 50, min_sections: int = 8) -> list[dict]:
    if not transcript:
        return []
    if max_items < 1:
        return []

    starts = [float(seg.get("start", 0.0)) for seg in transcript]
    durations = [float(seg.get("duration", 0.0)) for seg in transcript]
    video_start = starts[0]
    video_end = starts[-1] + (durations[-1] if durations else 0.0)
    total_duration = max(1.0, video_end - video_start)

    # Around 1 section per ~95s gives readable chunks for long-form videos.
    auto_sections = int(round(total_duration / 95.0))
    target_sections = max(min_sections, auto_sections)
    target_sections = min(target_sections, max_items, max(3, len(transcript) // 6))
    target_sections = max(1, target_sections)

    boundaries = [0]
    last_idx = 0
    for k in range(1, target_sections):
        desired_t = video_start + (total_duration * (k / target_sections))
        idx = next((j for j in range(last_idx + 1, len(transcript)) if starts[j] >= desired_t), len(transcript) - 1)
        lo = max(last_idx + 1, idx - 3)
        hi = min(len(transcript) - 1, idx + 3)
        for j in range(lo, hi + 1):
            prev_text = str(transcript[j - 1].get("text", "")) if j > 0 else ""
            if re.search(r"[.!?]$", prev_text.strip()):
                idx = j
                break
        if idx <= last_idx:
            continue
        boundaries.append(idx)
        last_idx = idx
    boundaries.append(len(transcript))

    items = []
    for sec_idx in range(len(boundaries) - 1):
        start_i = boundaries[sec_idx]
        end_i = boundaries[sec_idx + 1]
        if end_i <= start_i:
            continue
        raw = " ".join(str(seg.get("text", "")) for seg in transcript[start_i:end_i]).strip()
        n = len(items) + 1
        items.append({
            "n": n,
            "name": _infer_title_from_text(raw, index=n),
            "start": float(transcript[start_i].get("start", 0.0)),
            "seg_i": start_i
        })
    return items

def build_transcript_items(transcript: list[dict], max_items: int = 50, min_sections: int = 8) -> tuple[list[dict], str]:
    numbered = find_numbered_items(transcript, max_items=max_items)
    if _looks_like_numbered_list(numbered, max_items=max_items):
        items = _fill_small_gaps(numbered, max_items=max_items, max_gap=2)
        return items, "numbered_markers"

    time_items = build_time_sections(transcript, max_items=max_items, min_sections=min_sections)
    return time_items, "time_chunks"

def attach_section_text(transcript: list[dict], items: list[dict]) -> list[dict]:
    if not items:
        return items

    ordered = sorted(items, key=lambda x: float(x.get("start", 0.0)))
    starts = [float(seg.get("start", 0.0)) for seg in transcript]
    cursor = 0

    for i, it in enumerate(ordered):
        t0 = float(it.get("start", 0.0))
        t1 = float(ordered[i + 1]["start"]) if i + 1 < len(ordered) else float("inf")

        while cursor < len(transcript) and starts[cursor] < t0:
            cursor += 1
        j = cursor
        chunk_parts = []
        while j < len(transcript) and starts[j] < t1:
            chunk_parts.append(str(transcript[j].get("text", "")))
            j += 1
        it["raw_text"] = re.sub(r"\s+", " ", " ".join(chunk_parts)).strip()
        cursor = j

    ordered.sort(key=lambda x: int(x["n"]))
    return ordered


# ----------------------------
# Image scoring & extraction
# ----------------------------

_WARNED_NO_PILLOW = False
_WARNED_NO_TESSERACT = False
_TESSERACT_CMD: str | None = None
_TESSERACT_CHECKED = False
_OCR_CACHE: dict[tuple[str, int, int], str] = {}

def _resolve_tesseract_cmd() -> str | None:
    global _TESSERACT_CHECKED, _TESSERACT_CMD, _WARNED_NO_TESSERACT
    if _TESSERACT_CHECKED:
        return _TESSERACT_CMD

    candidates = [
        "tesseract",
        r"C:\Program Files\Tesseract-OCR\tesseract.exe",
        r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
        os.path.expandvars(r"%LOCALAPPDATA%\Programs\Tesseract-OCR\tesseract.exe"),
    ]

    # Try PATH/common install locations first.
    for cand in candidates:
        if not cand:
            continue
        try:
            r = subprocess.run([cand, "--version"], capture_output=True, text=False, timeout=4)
            if r.returncode == 0:
                _TESSERACT_CMD = cand
                _TESSERACT_CHECKED = True
                return _TESSERACT_CMD
        except Exception:
            pass

    # Last attempt: where.exe lookup (Windows).
    try:
        r = subprocess.run(["where.exe", "tesseract"], capture_output=True, text=False, timeout=4)
        if r.returncode == 0:
            out = (r.stdout or b"").decode("utf-8", errors="ignore")
            line = out.splitlines()
            if line:
                cand = line[0].strip()
                r2 = subprocess.run([cand, "--version"], capture_output=True, text=False, timeout=4)
                if r2.returncode == 0:
                    _TESSERACT_CMD = cand
    except Exception:
        pass

    _TESSERACT_CHECKED = True
    if _TESSERACT_CMD is None and not _WARNED_NO_TESSERACT:
        print("Warning: Tesseract not found. OCR keyword matching is disabled.")
        _WARNED_NO_TESSERACT = True
    return _TESSERACT_CMD

def _label_keywords(label: str) -> list[str]:
    raw = re.findall(r"[a-z0-9]+", (label or "").lower())
    if not raw:
        return []
    keep = []
    for w in raw:
        if (len(w) >= 3 and w not in STOP_WORDS) or w in {"ai", "ui", "seo", "pdf", "gpt", "api", "ocr"}:
            keep.append(w)
    if not keep:
        keep = [w for w in raw if len(w) >= 3][:3]

    # Add meaningful bigrams for stronger matches in OCR text.
    terms = []
    for w in keep:
        terms.append(w)
    for i in range(len(keep) - 1):
        a, b = keep[i], keep[i + 1]
        if len(a) >= 3 and len(b) >= 3:
            terms.append(f"{a} {b}")

    # Deduplicate preserving order.
    seen = set()
    out = []
    for t in terms:
        if t in seen:
            continue
        seen.add(t)
        out.append(t)
    return out[:12]

def _ocr_text(path: str) -> str:
    cmd = _resolve_tesseract_cmd()
    if not cmd:
        return ""
    try:
        st = Path(path).stat()
        key = (str(path), int(st.st_size), int(st.st_mtime_ns))
    except OSError:
        return ""

    cached = _OCR_CACHE.get(key)
    if cached is not None:
        return cached

    text_out = ""
    try:
        # psm 6: assume a block of text, good default for UI screenshots.
        r = subprocess.run(
            [cmd, str(path), "stdout", "--oem", "1", "--psm", "6", "-l", "eng"],
            capture_output=True,
            text=False,
            timeout=10,
        )
        if r.returncode == 0:
            text_out = (r.stdout or b"").decode("utf-8", errors="ignore")
    except Exception:
        text_out = ""

    norm = re.sub(r"[^a-z0-9]+", " ", text_out.lower())
    norm = re.sub(r"\s+", " ", norm).strip()
    _OCR_CACHE[key] = norm
    return norm

def _ocr_keyword_score(path: str, label: str) -> tuple[float, int]:
    text = _ocr_text(path)
    if not text:
        return 0.0, 0

    tokens = _label_keywords(label)
    if not tokens:
        return 0.0, 0

    words = set(text.split())
    compact_text = text.replace(" ", "")
    compact_label = "".join(re.findall(r"[a-z0-9]+", label.lower()))

    score = 0.0
    hits = 0
    for tok in tokens:
        if " " in tok:
            if tok in text:
                score += 9.0
                hits += 2
        else:
            if tok in words:
                score += 6.0
                hits += 1
            elif tok in compact_text:
                score += 3.0
                hits += 1

    if compact_label and len(compact_label) >= 6 and compact_label in compact_text:
        score += 12.0
        hits += 2

    # Fuzzy rescue for OCR typos on brand-like tokens.
    words = [w for w in text.split() if len(w) >= 4]
    for tok in tokens:
        if " " in tok or len(tok) < 6:
            continue
        best = 0.0
        for w in words:
            if abs(len(w) - len(tok)) > 2:
                continue
            r = SequenceMatcher(None, tok, w).ratio()
            if r > best:
                best = r
        if best >= 0.86:
            score += 4.0
            hits += 1
        elif best >= 0.80:
            score += 2.0
            hits += 1

    # Penalize readable-text frames that miss section keywords.
    if hits == 0 and len(words) >= 4:
        score -= 4.0

    return score, hits

def _file_size_score(path: str) -> float:
    try:
        return float(Path(path).stat().st_size) / 1024.0
    except OSError:
        return 0.0

def _stability_score(path_a: str, path_b: str) -> float:
    """
    Score temporal stability between two nearby frames.
    Higher means less motion/camera movement, which usually matches UI demos better.
    """
    try:
        from PIL import Image, ImageChops, ImageStat
    except Exception:
        return 0.0

    try:
        a = Image.open(path_a).convert("L").resize((320, 180))
        b = Image.open(path_b).convert("L").resize((320, 180))
        diff = ImageChops.difference(a, b)
        mean_diff = ImageStat.Stat(diff).mean[0]  # 0..255
        return max(0.0, 255.0 - float(mean_diff))
    except Exception:
        return 0.0

def _score_image(path: str) -> float:
    """
    Prefer 'screen-like' frames: sharp + text-rich + UI-like contrast + lower colorfulness.
    Falls back to file-size proxy if Pillow is unavailable.
    """
    global _WARNED_NO_PILLOW
    try:
        from PIL import Image, ImageFilter, ImageStat
    except Exception:
        if not _WARNED_NO_PILLOW:
            print("Warning: Pillow not installed. Frame scoring quality is reduced. Run: pip install -U pillow")
            _WARNED_NO_PILLOW = True
        return _file_size_score(path)

    try:
        img = Image.open(path).convert("RGB").resize((640, 360))
        gray = img.convert("L")
        stat_g = ImageStat.Stat(gray)
        mean = stat_g.mean[0]  # 0..255
        var = stat_g.var[0]

        edges = gray.filter(ImageFilter.FIND_EDGES)
        stat_e = ImageStat.Stat(edges)
        e_mean = stat_e.mean[0]
        e_var = stat_e.var[0]
        e_hist = edges.histogram()
        e_total = max(1, sum(e_hist))
        strong_edge_ratio = float(sum(e_hist[70:])) / float(e_total)

        hsv = img.convert("HSV")
        h, s_chan, v_chan = hsv.split()
        sat_mean = ImageStat.Stat(s_chan).mean[0]

        g_hist = gray.histogram()
        g_total = max(1, sum(g_hist))
        bright_ratio = float(sum(g_hist[210:])) / float(g_total)
        dark_ratio = float(sum(g_hist[:35])) / float(g_total)

        # Penalize center-heavy skin-like blobs (common for hand B-roll overlays).
        cx1, cy1, cx2, cy2 = 220, 120, 420, 300
        center = hsv.crop((cx1, cy1, cx2, cy2))
        skin_like = 0
        total_px = 0
        for hp, sp, vp in center.getdata():
            total_px += 1
            if (7 <= hp <= 30) and (35 <= sp <= 200) and (55 <= vp <= 255):
                skin_like += 1
        skin_ratio = (float(skin_like) / float(total_px)) if total_px else 0.0

        # Composite heuristic tuned for desktop/app UI frames.
        score = 0.0
        score += e_var * 0.9
        score += e_mean * 25.0
        score += var * 0.06
        score += strong_edge_ratio * 520.0
        score += bright_ratio * 140.0
        score += min(dark_ratio, 0.20) * 80.0
        score -= sat_mean * 0.75
        score -= skin_ratio * 260.0

        # Hard penalties for low-quality/non-UI frames.
        if mean < 35:
            score *= 0.30
        if bright_ratio < 0.08 and sat_mean > 95:
            score *= 0.60
        if strong_edge_ratio < 0.03:
            score *= 0.70
        return float(score)
    except Exception:
        return _file_size_score(path)

def _ffmpeg_frame(video_path: str, t_seconds: float, out_path: str) -> bool:
    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel", "error",
        "-ss", str(max(0.0, t_seconds)),
        "-i", video_path,
        "-frames:v", "1",
        "-q:v", "2",
        "-vf", "scale=1280:-1",
        out_path,
        "-y",
    ]
    try:
        subprocess.run(cmd, check=True)
        return True
    except Exception:
        return False

def _ffmpeg_gif(video_path: str, start_seconds: float, duration_seconds: float, out_path: str,
                fps: int = 8, width: int = 880) -> bool:
    start = max(0.0, float(start_seconds))
    duration = max(0.8, float(duration_seconds))
    fps = max(4, int(fps))
    width = max(320, int(width))

    vf = (
        f"fps={fps},scale={width}:-1:flags=lanczos,"
        "split[s0][s1];[s0]palettegen=max_colors=96[p];[s1][p]paletteuse"
    )
    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel", "error",
        "-ss", str(start),
        "-t", str(duration),
        "-i", video_path,
        "-vf", vf,
        "-loop", "0",
        out_path,
        "-y",
    ]
    try:
        subprocess.run(cmd, check=True)
        return True
    except Exception:
        return False

def extract_best_frame(video_path: str, base_time: float, out_path: str,
                       window_s: float = 10.0, candidates: int = 7,
                       segment_start: float | None = None,
                       segment_end: float | None = None,
                       label_hint: str | None = None,
                       ocr_top_k: int = 0,
                       use_stability: bool = True,
                       max_candidates: int = 28) -> tuple[bool, float, float, int]:
    """
    Extract multiple candidate frames in a bounded segment and keep the best-scoring one.
    """
    outp = Path(out_path)
    outp.parent.mkdir(parents=True, exist_ok=True)

    tmp_dir = outp.parent / "_candidates"
    tmp_dir.mkdir(exist_ok=True)

    if candidates < 2:
        candidates = 2

    search_start = max(0.0, base_time)
    search_end = search_start + max(0.8, window_s)
    if segment_end is not None and segment_end > search_start + 0.8:
        search_end = min(search_end, segment_end)
    if search_end <= search_start:
        search_end = search_start + 1.0

    span = max(0.8, search_end - search_start)
    adaptive_candidates = max(candidates, int(span / 1.35) + 3)
    adaptive_candidates = min(max_candidates, max(3, adaptive_candidates))

    times = [search_start + (span * (i / (adaptive_candidates - 1))) for i in range(adaptive_candidates)]
    scored_candidates: list[dict] = []

    tmp_files = []
    for i, t in enumerate(times):
        tmp = tmp_dir / f"cand_{outp.stem}_{i:02d}.jpg"
        if _ffmpeg_frame(video_path, t, str(tmp)):
            tmp_files.append(tmp)
            score = _score_image(str(tmp))

            if use_stability:
                # Temporal stability bonus (prefer steady UI over moving B-roll).
                probe_t = min(search_end, t + 0.35)
                if probe_t > t + 0.08:
                    probe = tmp_dir / f"probe_{outp.stem}_{i:02d}.jpg"
                    if _ffmpeg_frame(video_path, probe_t, str(probe)):
                        tmp_files.append(probe)
                        score += _stability_score(str(tmp), str(probe)) * 1.6

            if segment_start is not None and segment_end is not None and segment_end > segment_start + 1.0:
                pos = (t - segment_start) / (segment_end - segment_start)
                pos = max(0.0, min(1.0, pos))
                mid_bias = max(0.0, 1.0 - (abs(pos - 0.42) / 0.45))  # triangle peak near first half-middle
                score *= (1.0 + (mid_bias * 0.22))
                if pos < 0.12:
                    score *= 0.78

            scored_candidates.append({"path": tmp, "score": float(score), "t": float(t), "ocr_hits": 0})

    if not scored_candidates:
        return False, -1.0, search_start, 0

    # OCR rerank only for top visual candidates to keep runtime reasonable.
    if label_hint and ocr_top_k > 0 and _resolve_tesseract_cmd():
        top_n = min(max(1, ocr_top_k), len(scored_candidates))
        top_idx = sorted(range(len(scored_candidates)), key=lambda k: scored_candidates[k]["score"], reverse=True)[:top_n]
        for idx in top_idx:
            cand = scored_candidates[idx]
            bonus, hits = _ocr_keyword_score(str(cand["path"]), label_hint)
            if bonus > 0:
                cand["score"] += bonus
                cand["ocr_hits"] = hits
                scored_candidates[idx] = cand

    best = max(scored_candidates, key=lambda c: float(c["score"]))

    # move best to final, remove others
    try:
        shutil.move(str(best["path"]), str(outp))
    except Exception:
        try:
            shutil.copyfile(str(best["path"]), str(outp))
        except Exception:
            return False, -1.0, search_start, 0

    for p in tmp_files:
        if p.exists():
            try:
                p.unlink()
            except OSError:
                pass
    # attempt to remove tmp_dir if empty
    try:
        if tmp_dir.exists() and not any(tmp_dir.iterdir()):
            tmp_dir.rmdir()
    except OSError:
        pass

    return True, float(best["score"]), float(best["t"]), int(best.get("ocr_hits", 0))


# ----------------------------
# Markdown placeholder injection + HTML export
# ----------------------------

PLACEHOLDER_LINE_RE = re.compile(r"^\[IMAGE_PLACEHOLDER:\s*(?P<label>.+?)\s+at\s+timestamp\s+(?P<ts>\d{2}:\d{2})\]\s*$")

def build_transcript_article(items: list[dict], strategy: str) -> str:
    md = []
    md.append("# Transcript-driven Article")
    md.append("")
    md.append(
        f"Generated from transcript segmentation (`{strategy}`) and timestamped screenshot extraction."
    )
    md.append("")
    md.append("---")
    md.append("")
    for it in items:
        n = it["n"]
        name = it["name"]
        start = float(it["start"])
        cleaned = trim_section_lead_in(it.get("raw_text", ""), n)
        summary = short_summary(cleaned, 2) or "_(No transcript text captured for this section.)_"
        md.append(f"## {n}. {name}")
        md.append("")
        md.append(summary)
        md.append("")
        md.append(f"[IMAGE_PLACEHOLDER: {name} at timestamp {mmss(start)}]")
        md.append("")
        md.append("---")
        md.append("")
    return "\n".join(md).strip() + "\n"

def md_to_html_basic(md_text: str, style: str = "article") -> str:
    """Compatibility wrapper for the dedicated HTML renderer module."""
    return _md_to_html_basic_impl(md_text, style=style)


def inject_best_images(md_text: str, video_path: str, images_dir: str,
                       frame_offset: float, frame_window: float, candidates: int,
                       image_profile: str = "fast", verbose_picks: bool = False,
                       ocr_budget: int = 0, smart_retry: bool = False,
                       media_type: str = "frame", gif_duration: float = 2.6,
                       gif_fps: int = 8, gif_width: int = 880) -> tuple[str, int, int]:
    """
    Replace each placeholder line with a real image. Returns (new_md, replaced_count, total_placeholders).
    Profiles:
      - fast: single-pass visual scoring (default, low CPU)
      - balanced: visual scoring + early fallback
      - accurate: multi-pass + OCR reranking (slowest)
    Media:
      - frame: emit one JPG per section
      - gif: emit a short GIF clip per section
    """
    profile = (image_profile or "fast").strip().lower()
    if profile not in {"fast", "balanced", "accurate"}:
        profile = "fast"
    media = (media_type or "frame").strip().lower()
    if media not in {"frame", "gif"}:
        media = "frame"

    if profile == "fast":
        use_stability = False
        enable_early = False
        enable_mid = False
        candidate_cap = 8
        candidate_max = 10
        auto_ocr_budget = 0
    elif profile == "balanced":
        use_stability = True
        enable_early = True
        enable_mid = False
        candidate_cap = 12
        candidate_max = 14
        auto_ocr_budget = 0
    else:
        use_stability = True
        enable_early = True
        enable_mid = True
        candidate_cap = 20
        candidate_max = 24
        auto_ocr_budget = 10

    base_candidates = max(2, int(candidates))
    primary_candidates = min(candidate_cap, base_candidates)
    log_picks = verbose_picks or profile == "accurate"
    allow_targeted_ocr = (profile == "accurate") or bool(smart_retry)
    if ocr_budget > 0:
        remaining_ocr_budget = max(0, int(ocr_budget))
    elif allow_targeted_ocr and profile != "accurate":
        # Small default for fast/balanced smart retry.
        remaining_ocr_budget = 4 if profile == "fast" else 6
    else:
        remaining_ocr_budget = auto_ocr_budget

    lines = md_text.splitlines()
    out_lines = list(lines)
    replaced = 0
    placeholders = []

    for idx, line in enumerate(lines):
        m = PLACEHOLDER_LINE_RE.match(line.strip())
        if not m:
            continue
        label = m.group("label").strip()
        ts = m.group("ts")
        mm, ss = ts.split(":")
        t0 = int(mm) * 60 + int(ss)
        placeholders.append({"line_i": idx, "label": label, "ts": ts, "t0": t0})

    total = len(placeholders)

    for i, ph in enumerate(placeholders):
        label = ph["label"]
        ts = ph["ts"]
        t0 = float(ph["t0"])
        next_t = float(placeholders[i + 1]["t0"]) if i + 1 < len(placeholders) else None

        # Start searching shortly after marker; keep search inside this section when possible.
        base_time = max(0.0, t0 + frame_offset)
        segment_end = None
        effective_window = max(0.8, frame_window)

        if next_t is not None and next_t > t0:
            segment_end = max(base_time + 0.8, next_t - 0.25)
            effective_window = min(effective_window, max(0.8, segment_end - base_time))

            # If offset pushes us too close to next section, back off slightly.
            if segment_end - base_time < 1.0:
                base_time = max(0.0, t0 + max(0.6, frame_offset * 0.45))
                effective_window = min(frame_window, max(0.8, segment_end - base_time))

        ext = "gif" if media == "gif" else "jpg"
        fname = f"shot_{i + 1:02d}_{ts.replace(':','m')}s_{safe_slug(label)}.{ext}"
        out_path = os.path.join(images_dir, fname)
        stem = f"shot_{i + 1:02d}_{ts.replace(':','m')}s_{safe_slug(label)}"
        tmp_primary = os.path.join(images_dir, f"_tmp_primary_{stem}.jpg")
        tmp_early = os.path.join(images_dir, f"_tmp_early_{stem}.jpg")
        tmp_mid = os.path.join(images_dir, f"_tmp_mid_{stem}.jpg")
        tmp_ocr = os.path.join(images_dir, f"_tmp_ocr_{stem}.jpg")

        ok, score, chosen_t, _ = extract_best_frame(
            video_path,
            base_time,
            tmp_primary,
            window_s=effective_window,
            candidates=primary_candidates,
            segment_start=t0,
            segment_end=segment_end,
            label_hint=label,
            ocr_top_k=0,
            use_stability=use_stability,
            max_candidates=candidate_max,
        )

        chosen_tmp = tmp_primary if ok else None
        chosen_score = score if ok else -1.0
        chosen_time = chosen_t if ok else base_time

        # Fallback search near section start to catch UI that appears quickly after marker.
        if enable_early and frame_offset > 1.4:
            early_start = max(0.0, t0 + min(1.2, max(0.5, frame_offset * 0.35)))
            early_window = min(max(1.2, effective_window * 0.55), 8.0)
            ok2, score2, chosen_t2, _ = extract_best_frame(
                video_path,
                early_start,
                tmp_early,
                window_s=early_window,
                candidates=max(4, min(primary_candidates, primary_candidates // 2 + 2)),
                segment_start=t0,
                segment_end=segment_end,
                label_hint=label,
                ocr_top_k=0,
                use_stability=use_stability,
                max_candidates=candidate_max,
            )
            if ok2 and score2 > chosen_score:
                chosen_tmp = tmp_early
                chosen_score = score2
                chosen_time = chosen_t2

        # Mid/late pass for longer sections to avoid intro-only frames.
        if enable_mid and next_t is not None:
            section_span = max(0.0, next_t - t0)
            if section_span >= 10.0:
                late_anchor = t0 + max(frame_offset * 0.8, section_span * 0.35)
                late_start = max(base_time, min(late_anchor, next_t - 1.0))
                late_end = max(late_start + 0.8, next_t - 0.25)
                late_window = max(0.8, late_end - late_start)
                ok3, score3, chosen_t3, _ = extract_best_frame(
                    video_path,
                    late_start,
                    tmp_mid,
                    window_s=min(frame_window, late_window),
                    candidates=max(4, min(10, primary_candidates)),
                    segment_start=t0,
                    segment_end=late_end,
                    label_hint=label,
                    ocr_top_k=0,
                    use_stability=use_stability,
                    max_candidates=candidate_max,
                )
                if ok3 and (score3 * 1.04) > chosen_score:
                    chosen_tmp = tmp_mid
                    chosen_score = score3
                    chosen_time = chosen_t3

        # OCR is intentionally targeted: validate selected frame first,
        # then rerank only when it misses obvious label keywords.
        label_terms = _label_keywords(label)
        label_semantic = any(len(t) >= 6 and " " not in t for t in label_terms) or len(label_terms) >= 3
        if (
            allow_targeted_ocr
            and chosen_tmp
            and os.path.exists(chosen_tmp)
            and remaining_ocr_budget > 0
            and label_semantic
            and _resolve_tesseract_cmd()
        ):
            bonus_selected, hits_selected = _ocr_keyword_score(chosen_tmp, label)
            chosen_score += (bonus_selected * 0.55)
            remaining_ocr_budget -= 1

            if hits_selected == 0 and remaining_ocr_budget > 0:
                ocr_candidates = max(4, min(8, primary_candidates))
                ocr_window = min(frame_window, max(3.0, effective_window))
                ok4, score4, chosen_t4, hits4 = extract_best_frame(
                    video_path,
                    base_time,
                    tmp_ocr,
                    window_s=ocr_window,
                    candidates=ocr_candidates,
                    segment_start=t0,
                    segment_end=segment_end,
                    label_hint=label,
                    ocr_top_k=2,
                    use_stability=False,
                    max_candidates=12,
                )
                remaining_ocr_budget -= 1
                if ok4 and hits4 > 0 and score4 >= (chosen_score * 0.90):
                    chosen_tmp = tmp_ocr
                    chosen_score = score4
                    chosen_time = chosen_t4

        if chosen_tmp and os.path.exists(chosen_tmp):
            emitted = False
            if media == "gif":
                clip_len = max(1.0, float(gif_duration))
                clip_start = max(0.0, float(chosen_time) - min(0.8, clip_len * 0.25))
                if next_t is not None and next_t > clip_start:
                    clip_end = max(clip_start + 0.8, next_t - 0.10)
                    clip_len = min(clip_len, max(0.8, clip_end - clip_start))
                emitted = _ffmpeg_gif(
                    video_path,
                    clip_start,
                    clip_len,
                    out_path,
                    fps=gif_fps,
                    width=gif_width,
                )
            else:
                try:
                    shutil.move(chosen_tmp, out_path)
                    emitted = True
                except Exception:
                    try:
                        shutil.copyfile(chosen_tmp, out_path)
                        emitted = True
                    except Exception:
                        emitted = False

            if os.path.exists(tmp_primary):
                try:
                    os.remove(tmp_primary)
                except OSError:
                    pass
            if os.path.exists(tmp_early):
                try:
                    os.remove(tmp_early)
                except OSError:
                    pass
            if os.path.exists(tmp_mid):
                try:
                    os.remove(tmp_mid)
                except OSError:
                    pass
            if os.path.exists(tmp_ocr):
                try:
                    os.remove(tmp_ocr)
                except OSError:
                    pass

            if log_picks and next_t is not None:
                print(
                    f"  pick #{i + 1:02d} @ {mmss(chosen_time)} in [{mmss(t0)}..{mmss(max(t0, next_t - 0.25))}] "
                    f"label='{label}' score={chosen_score:.1f}"
                )

            if emitted:
                out_lines[ph["line_i"]] = f"![{label} (at {ts})](images/{fname})"
                replaced += 1

        # Best-effort temp cleanup even when extraction fails.
        for p in (tmp_primary, tmp_early, tmp_mid, tmp_ocr):
            if os.path.exists(p):
                try:
                    os.remove(p)
                except OSError:
                    pass

    return ("\n".join(out_lines) + "\n", replaced, total)


# ----------------------------
# Main pipeline
# ----------------------------

async def run_pipeline(
    url: str,
    output_dir: str,
    level: str = "A",
    mock: bool = False,
    model: str | None = None,
    max_items: int = 50,
    min_sections: int = 8,
    frame_offset: float = 3.5,
    frame_window: float = 10.0,
    candidates: int = 7,
    image_profile: str = "fast",
    ocr_budget: int = 0,
    smart_retry: bool = False,
    gif: bool = False,
    gif_duration: float = 2.6,
    gif_fps: int = 8,
    gif_width: int = 880,
    html_style: str = "article",
    mode: str = "transcript"
):
    print(f"--- Starting Pipeline for Level {level}  ---")

    from scripts.downloader import get_video_id, download_transcript, download_video
    from scripts.article_generator import generate_article

    # 1) Setup directories
    video_id = get_video_id(url) if not mock else "mock_video"
    work_dir = os.path.join(output_dir, video_id)
    os.makedirs(work_dir, exist_ok=True)

    images_dir = os.path.join(work_dir, "images")
    os.makedirs(images_dir, exist_ok=True)

    # 2) Transcript
    print("Step 1: Getting transcript...")
    transcript_path = os.path.join(work_dir, "transcript.json")
    if mock:
        src = os.path.join(_project_root(), "mock_transcript.json")
        shutil.copy(src, transcript_path)
    else:
        if not download_transcript(video_id, transcript_path):
            print("Failed to get transcript. Pipeline stopped.")
            return 1

    # 3) Article draft
    print("Step 2: Generating article draft...")
    article_draft_path = os.path.join(work_dir, "article_draft.md")

    if mode == "gemini":
        # Keep your existing Gemini writer (may produce fewer than 50 if prompt is weak)
        if not generate_article(transcript_path, article_draft_path, model=model):
            print("Failed to generate article. Pipeline stopped.")
            return 1
    else:
        # Transcript-driven: use numbered markers when available, otherwise time chunks.
        transcript = load_transcript(transcript_path)
        items, strategy = build_transcript_items(
            transcript,
            max_items=max_items,
            min_sections=min_sections,
        )
        if not items:
            print("Could not segment transcript into sections. Pipeline stopped.")
            return 1
        items = attach_section_text(transcript, items)
        print(f"Transcript strategy: {strategy} ({len(items)} sections)")
        md = build_transcript_article(items, strategy=strategy)
        Path(article_draft_path).write_text(md, encoding="utf-8")

    # 4) Images + final MD/HTML
    if level == "A":
        print("Step 3 (Level A): Downloading video (once) + injecting best-matching images...")
        print(f"Image picker profile: {image_profile}")
        print(f"Output media: {'GIF clips' if gif else 'single frames'}")
        print(f"HTML style: {html_style}")
        if image_profile == "accurate":
            print(f"OCR budget: {ocr_budget if ocr_budget > 0 else 'auto'}")
        elif smart_retry:
            print(f"Smart retry: on (OCR budget: {ocr_budget if ocr_budget > 0 else 'auto-small'})")
        video_path = download_video(url, work_dir) if not mock else None
        if not video_path or not os.path.exists(video_path):
            print("Failed to download video for image extraction.")
        else:
            md_text = Path(article_draft_path).read_text(encoding="utf-8", errors="replace")
            final_md, replaced, total = inject_best_images(
                md_text, video_path, images_dir,
                frame_offset=frame_offset,
                frame_window=frame_window,
                candidates=candidates,
                image_profile=image_profile,
                ocr_budget=ocr_budget,
                smart_retry=smart_retry,
                media_type="gif" if gif else "frame",
                gif_duration=gif_duration,
                gif_fps=gif_fps,
                gif_width=gif_width,
            )

            final_md_path = os.path.join(work_dir, "article_final.md")
            Path(final_md_path).write_text(final_md, encoding="utf-8")

            final_html_path = os.path.join(work_dir, "article_final.html")
            Path(final_html_path).write_text(md_to_html_basic(final_md, style=html_style), encoding="utf-8")

            print(f"Injected images: {replaced}/{total}")
            # Keep the video by default (useful when tuning offset/window)
            # If you want to auto-delete, uncomment:
            # try: os.remove(video_path)
            # except OSError: pass
    else:
        print("Level B not implemented in this patch.")

    print("\n--- Pipeline Complete! ---")
    print(f"Results are in: {work_dir}")
    print(f"Draft : {article_draft_path}")
    if level == "A":
        print(f"Final : {os.path.join(work_dir, 'article_final.md')}")
        print(f"HTML  : {os.path.join(work_dir, 'article_final.html')}")
        print(f"Images: {images_dir}")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="YouTube Video to Article Automation (Gemini + Correct Image Injection)")
    parser.add_argument("url", help="YouTube Video URL")
    parser.add_argument("--level", choices=["A", "B"], default="A", help="Automation level (A: Inject best-matching frames)")
    parser.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "output"), help="Output directory")
    parser.add_argument("--mock", action="store_true", help="Run in mock mode for testing")
    parser.add_argument("--model", default=None, help="Gemini model name (only used in --mode gemini)")
    parser.add_argument("--mode", choices=["transcript", "gemini"], default="transcript",
                        help="transcript = adaptive transcript segmentation; gemini = use scripts/article_generator output")
    parser.add_argument("--max-items", type=int, default=50, help="Maximum sections in transcript mode (default 50)")
    parser.add_argument("--min-sections", type=int, default=8, help="Minimum sections when falling back to time chunks")
    parser.add_argument("--frame-offset", type=float, default=3.5, help="Seconds AFTER the spoken number to start searching")
    parser.add_argument("--frame-window", type=float, default=10.0, help="How far forward to search for a good frame")
    parser.add_argument("--candidates", type=int, default=7, help="How many candidate frames to score in the window")
    parser.add_argument("--image-profile", choices=["fast", "balanced", "accurate"], default="fast",
                        help="fast=low CPU, balanced=better matching, accurate=OCR + multi-pass (slow)")
    parser.add_argument("--ocr-budget", type=int, default=0,
                        help="Only for --image-profile accurate: max sections to OCR-check/rerank (0=auto)")
    parser.add_argument("--smart-retry", action="store_true",
                        help="Enable lightweight targeted OCR retry for weak sections (works with fast/balanced)")
    parser.add_argument("--gif", action="store_true",
                        help="Emit short GIF clips instead of single JPG frames")
    parser.add_argument("--gif-duration", type=float, default=2.6,
                        help="GIF clip length in seconds (used only with --gif)")
    parser.add_argument("--gif-fps", type=int, default=8,
                        help="GIF frame rate (used only with --gif)")
    parser.add_argument("--gif-width", type=int, default=880,
                        help="GIF output width in pixels (used only with --gif)")
    parser.add_argument("--html-style", choices=["basic", "article"], default="article",
                        help="HTML style preset for article_final.html")
    parser.add_argument("--html-basic", action="store_true",
                        help="Shortcut for --html-style basic")
    parser.add_argument("--html-article", action="store_true",
                        help="Shortcut for --html-style article")

    args = parser.parse_args()
    if args.html_basic and args.html_article:
        parser.error("Use only one of --html-basic or --html-article")
    html_style = args.html_style
    if args.html_basic:
        html_style = "basic"
    elif args.html_article:
        html_style = "article"

    raise SystemExit(asyncio.run(run_pipeline(
        args.url, args.out, args.level, args.mock, args.model,
        max_items=args.max_items,
        min_sections=args.min_sections,
        frame_offset=args.frame_offset,
        frame_window=args.frame_window,
        candidates=args.candidates,
        image_profile=args.image_profile,
        ocr_budget=args.ocr_budget,
        smart_retry=args.smart_retry,
        gif=args.gif,
        gif_duration=args.gif_duration,
        gif_fps=args.gif_fps,
        gif_width=args.gif_width,
        html_style=html_style,
        mode=args.mode
    )))
