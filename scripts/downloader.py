import os
import sys
import subprocess
import shutil


def get_video_id(url: str) -> str:
    """Extract a YouTube video id from common URL formats."""
    if "v=" in url:
        return url.split("v=")[1].split("&")[0]
    if "youtu.be/" in url:
        return url.split("youtu.be/")[1].split("?")[0]
    return url


def download_transcript(video_id: str, output_path: str) -> bool:
    """Download YouTube transcript as JSON.

    Note: imports youtube_transcript_api lazily so mock mode can run without it.
    """
    try:
        from youtube_transcript_api import YouTubeTranscriptApi
        from youtube_transcript_api.formatters import JSONFormatter

        api = YouTubeTranscriptApi()
        transcript_list = api.list(video_id)
        # Prefer English if available, else fall back to first available.
        try:
            transcript_obj = transcript_list.find_transcript(["en"])
        except Exception:
            transcript_obj = next(iter(transcript_list))

        transcript = transcript_obj.fetch()
        formatter = JSONFormatter()
        json_formatted = formatter.format_transcript(transcript)

        with open(output_path, "w", encoding="utf-8") as f:
            f.write(json_formatted)
        return True
    except Exception as e:
        print(f"Error downloading transcript: {e}")
        return False


def download_video(url: str, output_dir: str) -> str | None:
    """Download a YouTube video (<=720p) using yt-dlp for frame extraction."""
    try:
        output_template = os.path.join(output_dir, "video.mp4")
        cmd = [
            "yt-dlp",
            "-f",
            "bestvideo[height<=720][ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best",
            "--merge-output-format",
            "mp4",
            url,
            "-o",
            output_template,
        ]
        subprocess.run(cmd, check=True)
        return output_template
    except Exception as e:
        print(f"Error downloading video: {e}")
        return None


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python3 downloader.py <url> <output_dir>")
        sys.exit(1)

    url = sys.argv[1]
    out_dir = sys.argv[2]
    vid = get_video_id(url)

    print(f"Processing video ID: {vid}")

    transcript_path = os.path.join(out_dir, "transcript.json")
    if download_transcript(vid, transcript_path):
        print(f"Transcript saved to {transcript_path}")

    video_path = download_video(url, out_dir)
    if video_path:
        print(f"Video saved to {video_path}")
