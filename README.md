# ClipCluster – AI Video Summarization and Scene Clustering System

![ClipCluster Architecture](https://img.shields.io/badge/Architecture-Modular%20CV%20%2B%20AI-6366f1)
![Python](https://img.shields.io/badge/Python-3.10%20--%203.14-10b981)
![Flask](https://img.shields.io/badge/Backend-Flask%203.x-f59e0b)
![OpenCV](https://img.shields.io/badge/Vision-OpenCV%205.x-ec4899)
![PyTorch](https://img.shields.io/badge/Deep%20Learning-Torch%20%2B%20Torchvision-0ea5e9)
![Scikit-Learn](https://img.shields.io/badge/Clustering-K--Means-8b5cf6)
![yt-dlp](https://img.shields.io/badge/YouTube-yt--dlp%20Stream-ff0033)
![FFmpeg](https://img.shields.io/badge/Encoder-FFmpeg%20H.264-14b8a6)

**ClipCluster** is an AI-powered computer vision and machine learning web application that automatically analyzes long video content from **local uploads** or **YouTube video links**, identifies important scenes, clusters visually and semantically related shots, and synthesizes a concise, playable video summary tailored to user-specified duration targets (25%, 50%, 75%, or custom percentage).

---

## Key Features

### 1. Dual Video Input Modes
- **Option 1: Upload Video File**:
  - Direct drag-and-drop or file picker from your computer.
  - Supports MP4, MOV, AVI, MKV, and WebM formats up to 500 MB.
  - Instant HTML5 video preview player with duration, resolution, FPS, and file size inspection.
- **Option 2: YouTube Video Link (NEW)**:
  - Paste any standard YouTube watch link (`youtube.com/watch?v=...`), shortened link (`youtu.be/...`), or YouTube Shorts link (`youtube.com/shorts/...`).
  - Automatic URL validation and access permission verification (respects copyright, DRM, age-gated, and private restrictions).
  - Retrieves real-time video title, channel name, duration, and thumbnail preview.
  - One-click **"Analyze YouTube Video"** button streams authorized videos directly into the ClipCluster AI pipeline.
- **Quick Demo Sample Video**:
  - Built-in 24-second, 6-scene test video with distinct environments, motion dynamics, and timecodes for instant testing without finding a video file.

### 2. Adaptive Scene Boundary Detection
- Content-aware inter-frame difference computation combining **3D HSV color histograms** (Bhattacharyya distance) and **pixel luminance change**.
- Dynamic thresholding ($\mu + k \cdot \sigma$) with minimum scene length constraints to prevent over-segmentation.
- Multi-metric optimal keyframe selection using **Laplacian Sharpness Variance** ($\text{Var}(\nabla^2 I)$) and lighting exposure balance.

### 3. Multi-Modal Visual Feature Extraction & Semantics
- Generates **500+ dimensional normalized visual embedding vectors**.
- **PyTorch MobileNetV3** deep embeddings + OpenCV color moments, Sobel edge direction gradients, and motion dynamics.
- Automatic **dominant color extraction** using mini-batch K-Means (top 3 colors with HEX codes and color names).
- Natural language **semantic scene explanations** (environment setting, camera activity level, visual focus).

### 4. K-Means Scene Clustering
- Groups similar scenes into cohesive visual themes using **Scikit-Learn K-Means**.
- Automatic cluster count estimation using **Silhouette score** analysis or customizable cluster selection ($k=2$ to $5$).
- Automatic cluster labeling (e.g., *Outdoor Natural Landscape*, *Warm Architecture*, *Dynamic High-Action*).
- Cluster exemplar selection and interactive filter pill tabs.

### 5. AI Video Summarization, Full Audio Synchronization & H.264 Encoding
- Diversity-aware knapsack scene selection algorithm: guarantees every cluster is represented, prioritizes high-sharpness and dynamic action scenes, and eliminates repetitive redundant takes.
- Summary duration control: **25% (Highlights)**, **50% (Balanced)**, **75% (Detailed Overview)**, or custom percentage slider.
- **Full Audio Stream Preservation**: Synchronized AAC stereo audio with monotonic presentation timestamps (PTS) and faststart headers, guaranteeing pristine sound in all web browsers and media players.
- FFmpeg H.264 (`libx264` / `yuv420p`) compilation guaranteeing universal, zero-glitch browser playback and download.

### 6. AI Short & Simple Video Notes
- **Executive Quick Synopsis**: Concise paragraph summarizing video duration savings and key environmental settings.
- **Key Takeaways & Bullet Points**: Easy-to-read bullet points highlighting essential stats, time savings, and themes.
- **Interactive Scene Highlights**: Timecode-referenced breakdown with 1-click **"▶ Jump"** buttons that seek video playback directly to that moment.
- **Copy & Download**: One-click **"Copy Notes"** to clipboard or download formatted Markdown report (`.md`).

### 7. Topic-Specific Video Condensation (NEW)
- **Search Any Topic**: Enter custom queries (e.g., *Forest*, *Sunset*, *Dynamic Action*, *Water*, *Clean Studio*) to condense and isolate only the specific part of the video covering that subject.
- **Smart Suggested Topics**: Auto-suggests topic tags based on detected themes and environments in the video.
- **Dedicated Topic Video Player**: Plays the focused topic clip with audio enabled.
- **Topic-Specific Notes & Matched Scenes**: Displays concise topic takeaways and clickable segment tags.
- **Download Topic Video & Notes**: Instant download of the isolated MP4 clip and topic notes.

### 8. Interactive Results Dashboard
- Dual video player comparison (AI-Summarized Video vs. Full Original / YouTube Video).
- Interactive Color-Coded Scene Timeline bar (click any segment to jump video playback).
- Key Stat cards: Original duration, Summary duration, Duration reduction %, Scene count, Cluster count.
- Download options: Summarized MP4 video, Short Notes (.md), Topic Video (.mp4), Topic Notes (.md), formatted JSON intelligence report, and CSV scene table.

---

## Directory Structure

```
clipcluster/
├── .vscode/
│   ├── launch.json            # VS Code Run & Debug configuration (F5 key support)
│   └── settings.json          # Pre-configured Python interpreter & environment
├── core/
│   ├── __init__.py
│   ├── video_processor.py     # Video inspection and container metadata
│   ├── youtube_handler.py     # YouTube link validation, metadata & authorized retrieval
│   ├── scene_detector.py      # Adaptive scene boundary detection & keyframe extraction
│   ├── feature_extractor.py   # Deep embeddings (PyTorch/MobileNet) + CV descriptors & semantics
│   ├── clusterer.py           # K-Means clustering, cluster naming & silhouette analysis
│   ├── summarizer.py          # Diversity knapsack selection & FFmpeg H.264 compiler
│   └── sample_generator.py    # Multi-scene test video synthesizer (6 distinct scenes)
├── static/
│   ├── css/
│   │   └── style.css          # Responsive dark-mode presentation UI
│   ├── js/
│   │   └── app.js             # Tabs, drag & drop, YouTube preview, live polling, player
│   ├── sample/                # Demo test video
│   ├── thumbnails/            # Extracted scene thumbnails
│   └── uploads/               # Staged uploads and compiled summaries
├── templates/
│   └── index.html             # Single-page application dashboard
├── app.py                     # Flask web server & REST API endpoints
├── config.py                  # Directories, upload limits, and FFmpeg configuration
├── open_in_vscode.bat         # 1-click batch script to open in VS Code
├── requirements.txt           # Python dependencies
├── run_clipcluster.bat        # Windows 1-click launcher script
└── test_suite.py              # Automated test suite
```

---

## How to Run the Application Locally

### Method 1: In VS Code (F5 Support)
1. Open VS Code: Click **File** → **Open Folder...** and select:
   `C:\Users\SAMEEKSHA SHETTY\.gemini\antigravity\scratch\clipcluster`
2. Press **`F5`** (or open the terminal with `Ctrl + ~` and type `python app.py`).
3. Open `http://127.0.0.1:5000` in your browser.

### Method 2: 1-Click Windows Batch Script
Navigate to `C:\Users\SAMEEKSHA SHETTY\.gemini\antigravity\scratch\clipcluster` and double-click:
```bat
run_clipcluster.bat
```

---

## YouTube Processing & Permissions Handling

When a user pastes a YouTube URL:
1. **Validation**: Checks for valid `youtube.com/watch?v=...`, `youtu.be/...`, or `youtube.com/shorts/...` format.
2. **Access Control**: Respects YouTube access restrictions. If a video is private, paywalled, DRM-protected, or requires login, ClipCluster catches this immediately and displays an alert prompting the user to upload an authorized local copy instead.
3. **Seamless Pipeline**: Once retrieved, the video flows through the exact same OpenCV scene detection, PyTorch feature extraction, K-Means clustering, and FFmpeg summarization as local files.
