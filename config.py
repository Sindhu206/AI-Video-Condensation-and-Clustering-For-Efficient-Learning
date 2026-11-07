import os
import shutil
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

STATIC_DIR = os.path.join(BASE_DIR, "static")
UPLOAD_DIR = os.path.join(STATIC_DIR, "uploads")
THUMBNAIL_DIR = os.path.join(STATIC_DIR, "thumbnails")
SAMPLE_DIR = os.path.join(STATIC_DIR, "sample")

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(THUMBNAIL_DIR, exist_ok=True)
os.makedirs(SAMPLE_DIR, exist_ok=True)

MAX_CONTENT_LENGTH = 500 * 1024 * 1024  # 500 MB limit
ALLOWED_EXTENSIONS = {"mp4", "mov", "avi", "mkv", "webm", "m4v"}

def get_ffmpeg_binary() -> str:
    """Find FFmpeg binary using imageio_ffmpeg or system PATH."""
    try:
        import imageio_ffmpeg
        exe = imageio_ffmpeg.get_ffmpeg_exe()
        if exe and os.path.exists(exe):
            return exe
    except Exception:
        pass
    
    system_ffmpeg = shutil.which("ffmpeg")
    if system_ffmpeg:
        return system_ffmpeg
    
    return "ffmpeg"
