import os
import uuid
import threading
import time
import json
import csv
import io
import shutil
from flask import Flask, render_template, request, jsonify, send_file, url_for
from werkzeug.utils import secure_filename

import config
from core.video_processor import VideoProcessor
from core.scene_detector import SceneDetector
from core.feature_extractor import FeatureExtractor, init_shared_torch_model
from core.clusterer import SceneClusterer
from core.summarizer import VideoSummarizer
from core.sample_generator import create_sample_video
from core.youtube_handler import YouTubeHandler, is_valid_youtube_url
from core.notes_generator import NotesGenerator
from core.topic_condenser import TopicCondenser
import re

app = Flask(__name__, static_folder="static", template_folder="templates")
app.config["MAX_CONTENT_LENGTH"] = config.MAX_CONTENT_LENGTH

# In-memory dictionary for tracking asynchronous background processing tasks
TASKS = {}
TASKS_LOCK = threading.Lock()

def update_task_progress(task_id: str, progress: int, message: str):
    """Thread-safe update of task progress."""
    with TASKS_LOCK:
        if task_id in TASKS:
            TASKS[task_id]["progress"] = progress
            TASKS[task_id]["message"] = message

def run_pipeline_task(
    task_id: str,
    video_path: str = None,
    summary_pct: float = 50.0,
    num_clusters: int = 0,
    youtube_url: str = None
):
    """Background worker executing the complete video summarization & clustering pipeline."""
    try:
        task_dir = os.path.join(config.UPLOAD_DIR, task_id)
        thumb_dir = os.path.join(config.THUMBNAIL_DIR, task_id)
        os.makedirs(task_dir, exist_ok=True)
        os.makedirs(thumb_dir, exist_ok=True)

        # Handle YouTube download if requested
        if youtube_url:
            update_task_progress(task_id, 8, "Validating access and obtaining authorized YouTube stream...")
            video_path = YouTubeHandler.download_video(
                youtube_url,
                task_dir,
                progress_callback=lambda p, msg: update_task_progress(task_id, p, msg)
            )

        if not video_path or not os.path.exists(video_path):
            raise FileNotFoundError("Video source file is unavailable for processing.")

        update_task_progress(task_id, 32, "Inspecting video format and audio streams...")
        metadata = VideoProcessor.get_video_metadata(video_path)
        
        # Keep web-accessible preview URL
        filename = os.path.basename(video_path)
        metadata["preview_url"] = f"/static/uploads/{task_id}/{filename}"
        
        # If this originated from YouTube, attach YouTube title & channel
        with TASKS_LOCK:
            yt_meta = TASKS.get(task_id, {}).get("youtube_meta")
            if yt_meta:
                metadata["youtube_title"] = yt_meta.get("title")
                metadata["youtube_channel"] = yt_meta.get("channel")
                metadata["youtube_embed_url"] = yt_meta.get("embed_url")

        # Scene Detection
        update_task_progress(task_id, 38, "Running adaptive scene boundary detection...")
        detector = SceneDetector(min_scene_sec=1.5, threshold_factor=2.0)
        scenes = detector.detect_scenes(
            video_path,
            thumb_dir,
            task_id,
            progress_callback=lambda p, msg: update_task_progress(task_id, p, msg)
        )

        if not scenes:
            raise ValueError("No valid scenes could be detected in this video.")

        # Feature Extraction & Semantics
        update_task_progress(task_id, 52, "Extracting visual embeddings, palettes & semantic descriptions...")
        fe = FeatureExtractor(use_deep_model=True)
        feature_matrix = fe.process_scenes_features(
            scenes,
            progress_callback=lambda p, msg: update_task_progress(task_id, p, msg)
        )

        # Scene Clustering
        update_task_progress(task_id, 68, "Executing K-Means clustering & cluster synthesis...")
        clusterer = SceneClusterer()
        scenes, cluster_info = clusterer.cluster_scenes(
            scenes,
            feature_matrix,
            num_clusters=num_clusters if num_clusters > 0 else None
        )

        # Video Summarization
        update_task_progress(task_id, 75, "Selecting optimal diverse key segments for summary...")
        summarizer = VideoSummarizer()
        selected_scenes = summarizer.select_summary_scenes(
            scenes,
            target_percentage=summary_pct,
            cluster_info=cluster_info
        )

        update_task_progress(task_id, 82, "Compiling summary video with audio preservation...")
        summary_filename = f"summary_{task_id}.mp4"
        summary_video_path = os.path.join(task_dir, summary_filename)
        
        summary_res = summarizer.generate_summary_video(
            video_path,
            selected_scenes,
            summary_video_path,
            progress_callback=lambda p, msg: update_task_progress(task_id, p, msg)
        )

        # Metrics calculation
        orig_dur = metadata["duration_sec"]
        summ_dur = summary_res["summary_duration_sec"]
        reduction_pct = round(((orig_dur - summ_dur) / max(0.1, orig_dur)) * 100.0, 1) if orig_dur > 0 else 0.0

        # Relative URLs for frontend
        summary_rel_url = f"/static/uploads/{task_id}/{summary_filename}"
        
        summary_info_dict = {
            "url": summary_rel_url,
            "file_path": summary_video_path,
            "duration_sec": summ_dur,
            "duration_formatted": summary_res["summary_duration_formatted"],
            "reduction_percentage": max(0.0, reduction_pct),
            "selected_scenes_count": len(selected_scenes),
            "target_percentage": summary_pct,
            "has_audio": summary_res.get("has_audio", False)
        }

        update_task_progress(task_id, 96, "Generating AI Executive Summary & Short Notes...")
        short_notes = NotesGenerator.generate_short_notes(
            metadata=metadata,
            all_scenes=scenes,
            selected_scenes=selected_scenes,
            cluster_info=cluster_info,
            summary_info=summary_info_dict
        )

        suggested_topics = TopicCondenser.extract_suggested_topics(scenes, cluster_info)

        results_payload = {
            "task_id": task_id,
            "metadata": metadata,
            "scenes": scenes,
            "clusters": cluster_info["clusters"],
            "total_clusters": cluster_info["total_clusters"],
            "silhouette_score": cluster_info["silhouette"],
            "summary": summary_info_dict,
            "short_notes": short_notes,
            "suggested_topics": suggested_topics,
            "stats": {
                "original_duration_formatted": metadata["duration_formatted"],
                "original_duration_sec": orig_dur,
                "summary_duration_formatted": summary_res["summary_duration_formatted"],
                "summary_duration_sec": summ_dur,
                "total_scenes": len(scenes),
                "total_clusters": cluster_info["total_clusters"],
                "reduction_percentage": max(0.0, reduction_pct),
                "has_audio": summary_res.get("has_audio", False)
            }
        }

        with TASKS_LOCK:
            TASKS[task_id]["status"] = "completed"
            TASKS[task_id]["progress"] = 100
            TASKS[task_id]["message"] = "Processing completed successfully!"
            TASKS[task_id]["results"] = results_payload
            TASKS[task_id]["video_path"] = video_path
            TASKS[task_id]["topic_results"] = {}

    except Exception as e:
        import traceback
        err_msg = str(e)
        traceback.print_exc()
        with TASKS_LOCK:
            TASKS[task_id]["status"] = "error"
            TASKS[task_id]["message"] = f"Processing error: {err_msg}"
            TASKS[task_id]["error"] = err_msg

@app.route("/")
def index():
    """Render the primary dashboard."""
    return render_template("index.html")

@app.route("/api/sample-video", methods=["GET"])
def get_sample_video():
    """Provide instant sample test video metadata without manual upload."""
    try:
        sample_path = os.path.join(config.SAMPLE_DIR, "demo_sample_video.mp4")
        if not os.path.exists(sample_path):
            sample_path = create_sample_video(sample_path)
            
        task_id = str(uuid.uuid4())
        task_dir = os.path.join(config.UPLOAD_DIR, task_id)
        os.makedirs(task_dir, exist_ok=True)
        
        # Copy to task directory
        dest_path = os.path.join(task_dir, "demo_sample_video.mp4")
        shutil.copyfile(sample_path, dest_path)
        
        meta = VideoProcessor.get_video_metadata(dest_path)
        meta["preview_url"] = f"/static/uploads/{task_id}/demo_sample_video.mp4"
        meta["task_id"] = task_id
        meta["source_type"] = "sample"
        
        with TASKS_LOCK:
            TASKS[task_id] = {
                "status": "ready",
                "source_type": "sample"
            }
            
        return jsonify({"success": True, "metadata": meta, "task_id": task_id})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

@app.route("/api/upload", methods=["POST"])
def upload_video():
    """Handle video file upload and return initial video metadata & preview URL."""
    if "video" not in request.files:
        return jsonify({"success": False, "error": "No video file found in upload request."}), 400
        
    file = request.files["video"]
    if file.filename == "":
        return jsonify({"success": False, "error": "No video file selected."}), 400
        
    filename = secure_filename(file.filename)
    ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    
    if ext not in config.ALLOWED_EXTENSIONS:
        return jsonify({
            "success": False,
            "error": f"Unsupported format (.{ext}). Allowed formats: {', '.join(config.ALLOWED_EXTENSIONS)}"
        }), 400

    task_id = str(uuid.uuid4())
    task_dir = os.path.join(config.UPLOAD_DIR, task_id)
    os.makedirs(task_dir, exist_ok=True)
    
    video_save_path = os.path.join(task_dir, filename)
    file.save(video_save_path)
    
    try:
        meta = VideoProcessor.get_video_metadata(video_save_path)
        meta["preview_url"] = f"/static/uploads/{task_id}/{filename}"
        meta["task_id"] = task_id
        meta["source_type"] = "upload"
        
        with TASKS_LOCK:
            TASKS[task_id] = {
                "status": "ready",
                "source_type": "upload"
            }
            
        return jsonify({"success": True, "metadata": meta, "task_id": task_id})
    except Exception as e:
        return jsonify({"success": False, "error": f"Failed to analyze uploaded video: {str(e)}"}), 400

@app.route("/api/youtube-info", methods=["POST"])
def youtube_info():
    """Retrieve YouTube video metadata, thumbnail, and validate access permissions."""
    data = request.get_json() or {}
    url = data.get("url", "").strip()

    if not url:
        return jsonify({"success": False, "error": "Please provide a YouTube video URL."}), 400

    if not is_valid_youtube_url(url):
        return jsonify({
            "success": False,
            "error": "Invalid YouTube URL format. Supported: youtube.com/watch?v=..., youtu.be/..., and youtube.com/shorts/..."
        }), 400

    try:
        meta = YouTubeHandler.get_metadata(url)
        task_id = str(uuid.uuid4())
        task_dir = os.path.join(config.UPLOAD_DIR, task_id)
        os.makedirs(task_dir, exist_ok=True)

        meta["task_id"] = task_id
        meta["source_type"] = "youtube"

        with TASKS_LOCK:
            TASKS[task_id] = {
                "status": "ready",
                "source_type": "youtube",
                "youtube_url": url,
                "youtube_meta": meta
            }

        return jsonify({"success": True, "task_id": task_id, "metadata": meta})
    except PermissionError as pe:
        return jsonify({
            "success": False,
            "error": str(pe),
            "is_permission_error": True
        }), 403
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e),
            "is_permission_error": False
        }), 400

@app.route("/api/process", methods=["POST"])
def start_processing():
    """Start the AI analysis and summarization pipeline in the background."""
    data = request.get_json() or {}
    task_id = data.get("task_id")
    summary_pct = float(data.get("summary_percentage", 50.0))
    num_clusters = int(data.get("num_clusters", 0))

    if not task_id:
        return jsonify({"success": False, "error": "Missing task_id."}), 400

    task_dir = os.path.join(config.UPLOAD_DIR, task_id)
    if not os.path.exists(task_dir):
        return jsonify({"success": False, "error": "Task directory does not exist. Please re-upload or re-enter the link."}), 404

    with TASKS_LOCK:
        task_info = TASKS.get(task_id, {})
        youtube_url = task_info.get("youtube_url")

    video_path = None
    if not youtube_url:
        video_files = [f for f in os.listdir(task_dir) if not f.startswith("summary_") and any(f.endswith(f".{e}") for e in config.ALLOWED_EXTENSIONS)]
        if not video_files:
            return jsonify({"success": False, "error": "No source video file found for this task."}), 404
        video_path = os.path.join(task_dir, video_files[0])

    with TASKS_LOCK:
        TASKS[task_id].update({
            "status": "processing",
            "progress": 5,
            "message": "Initializing video processing pipeline...",
            "error": None,
            "results": None
        })

    thread = threading.Thread(
        target=run_pipeline_task,
        args=(task_id, video_path, summary_pct, num_clusters, youtube_url),
        daemon=True
    )
    thread.start()

    return jsonify({"success": True, "task_id": task_id, "message": "Processing started."})

@app.route("/api/status/<task_id>", methods=["GET"])
def check_status(task_id: str):
    """Poll task status and live progress."""
    with TASKS_LOCK:
        task = TASKS.get(task_id)
        if not task:
            return jsonify({"status": "not_found", "message": "Task not found."}), 404
            
        return jsonify({
            "status": task.get("status"),
            "progress": task.get("progress", 0),
            "message": task.get("message", ""),
            "error": task.get("error")
        })

@app.route("/api/results/<task_id>", methods=["GET"])
def get_results(task_id: str):
    """Retrieve complete results after pipeline finishes."""
    with TASKS_LOCK:
        task = TASKS.get(task_id)
        if not task:
            return jsonify({"success": False, "error": "Task not found."}), 404
            
        if task.get("status") != "completed":
            return jsonify({"success": False, "status": task.get("status"), "error": "Processing not completed yet."}), 400
            
        return jsonify({"success": True, "data": task.get("results")})

@app.route("/api/condense-topic", methods=["POST"])
def condense_topic():
    """
    On-demand condensation of a specific topic from the analyzed video.
    Finds the exact matching video segments for the user's entered topic,
    compiles a focused video clip with audio, and returns topic notes.
    """
    data = request.get_json() or {}
    task_id = data.get("task_id")
    topic = data.get("topic", "").strip()

    if not task_id:
        return jsonify({"success": False, "error": "Missing task_id."}), 400

    if not topic:
        return jsonify({"success": False, "error": "Please enter a topic to condense."}), 400

    with TASKS_LOCK:
        task = TASKS.get(task_id)
        if not task or not task.get("results"):
            return jsonify({"success": False, "error": "Task results not found. Please run the video analysis first."}), 404
        results = task["results"]
        video_path = task.get("video_path")

    # If video_path is missing from task dict, locate it in task upload dir
    task_dir = os.path.join(config.UPLOAD_DIR, task_id)
    if not video_path or not os.path.exists(video_path):
        video_files = [f for f in os.listdir(task_dir) if not f.startswith("summary_") and not f.startswith("topic_") and any(f.endswith(f".{e}") for e in config.ALLOWED_EXTENSIONS)]
        if not video_files:
            return jsonify({"success": False, "error": "Source video file unavailable for topic condensation."}), 404
        video_path = os.path.join(task_dir, video_files[0])

    try:
        condenser = TopicCondenser()
        scenes = results.get("scenes", [])
        cluster_info = {"clusters": results.get("clusters", [])}
        metadata = results.get("metadata", {})

        matched_scenes, scored_scenes, match_meta = condenser.match_scenes(
            topic,
            scenes,
            cluster_info=cluster_info
        )

        if not matched_scenes:
            return jsonify({
                "success": False,
                "error": f"No scenes in this video matched the topic '{topic}'."
            }), 404

        # Compile topic video
        safe_topic_slug = re.sub(r'[^a-zA-Z0-9]', '_', topic.lower()).strip('_')[:20] or "focus"
        topic_filename = f"topic_{safe_topic_slug}_{task_id[:8]}.mp4"
        topic_video_path = os.path.join(task_dir, topic_filename)

        video_res = condenser.condense_topic_video(
            video_path,
            matched_scenes,
            topic_video_path
        )

        # Generate topic notes
        topic_notes = NotesGenerator.generate_topic_notes(
            topic=topic,
            matched_scenes=matched_scenes,
            metadata=metadata,
            match_meta=match_meta
        )

        topic_url = f"/static/uploads/{task_id}/{topic_filename}"

        topic_payload = {
            "topic": topic,
            "video_url": topic_url,
            "duration_sec": video_res["summary_duration_sec"],
            "duration_formatted": video_res["summary_duration_formatted"],
            "matched_scenes_count": len(matched_scenes),
            "total_scenes_count": len(scenes),
            "matched_scenes": matched_scenes,
            "match_meta": match_meta,
            "topic_notes": topic_notes,
            "has_audio": video_res.get("has_audio", True)
        }

        with TASKS_LOCK:
            if "topic_results" not in TASKS[task_id]:
                TASKS[task_id]["topic_results"] = {}
            TASKS[task_id]["topic_results"][topic.lower()] = topic_payload

        return jsonify({
            "success": True,
            "data": topic_payload
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"success": False, "error": f"Failed to condense topic: {str(e)}"}), 500

@app.route("/api/export/<task_id>/topic-notes", methods=["GET"])
def export_topic_notes(task_id: str):
    """Download specific Topic Notes report as a Markdown document."""
    topic = request.args.get("topic", "").strip().lower()
    with TASKS_LOCK:
        task = TASKS.get(task_id)
        if not task or not task.get("topic_results"):
            return jsonify({"error": "No topic results found for this task."}), 404
        
        topic_results = task["topic_results"]
        if topic and topic in topic_results:
            topic_data = topic_results[topic]
        elif topic_results:
            topic_data = next(iter(topic_results.values()))
        else:
            return jsonify({"error": "Topic notes not found."}), 404
            
        full_text = topic_data.get("topic_notes", {}).get("full_text", "No topic notes available.")
        topic_name = topic_data.get("topic", "topic")
        safe_topic = re.sub(r'[^a-zA-Z0-9]', '_', topic_name.lower())
        
    mem = io.BytesIO()
    mem.write(full_text.encode("utf-8"))
    mem.seek(0)
    return send_file(
        mem,
        mimetype="text/markdown",
        as_attachment=True,
        download_name=f"clipcluster_topic_notes_{safe_topic}_{task_id[:8]}.md"
    )

@app.route("/api/export/<task_id>/notes", methods=["GET"])
def export_notes(task_id: str):
    """Download full Short Notes report as a Markdown/Text document."""
    with TASKS_LOCK:
        task = TASKS.get(task_id)
        if not task or not task.get("results"):
            return jsonify({"error": "Task results not found."}), 404
        short_notes = task["results"].get("short_notes", {})
        full_text = short_notes.get("full_text", "No short notes available.")
        
    mem = io.BytesIO()
    mem.write(full_text.encode("utf-8"))
    mem.seek(0)
    return send_file(
        mem,
        mimetype="text/markdown",
        as_attachment=True,
        download_name=f"clipcluster_summary_notes_{task_id[:8]}.md"
    )

@app.route("/api/export/<task_id>/json", methods=["GET"])
def export_json(task_id: str):
    """Download full analysis report as a JSON file."""
    with TASKS_LOCK:
        task = TASKS.get(task_id)
        if not task or not task.get("results"):
            return jsonify({"error": "Task results not found."}), 404
        results = task["results"]
        
    mem = io.BytesIO()
    mem.write(json.dumps(results, indent=2).encode("utf-8"))
    mem.seek(0)
    return send_file(
        mem,
        mimetype="application/json",
        as_attachment=True,
        download_name=f"clipcluster_report_{task_id[:8]}.json"
    )

@app.route("/api/export/<task_id>/csv", methods=["GET"])
def export_csv(task_id: str):
    """Download scene breakdown table as a CSV file."""
    with TASKS_LOCK:
        task = TASKS.get(task_id)
        if not task or not task.get("results"):
            return jsonify({"error": "Task results not found."}), 404
        scenes = task["results"].get("scenes", [])
        
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow([
        "Scene ID", "Start Time", "End Time", "Duration (sec)",
        "Cluster ID", "In Summary", "Setting", "Activity", "Sharpness", "Motion Score", "Explanation"
    ])
    
    for s in scenes:
        sem = s.get("semantic", {})
        writer.writerow([
            s.get("scene_id"),
            s.get("start_time_str"),
            s.get("end_time_str"),
            s.get("duration_sec"),
            s.get("cluster_id"),
            "Yes" if s.get("is_in_summary") else "No",
            sem.get("setting", ""),
            sem.get("activity", ""),
            s.get("sharpness_score"),
            s.get("motion_score"),
            sem.get("explanation", "")
        ])
        
    mem = io.BytesIO()
    mem.write(output.getvalue().encode("utf-8"))
    mem.seek(0)
    return send_file(
        mem,
        mimetype="text/csv",
        as_attachment=True,
        download_name=f"clipcluster_scenes_{task_id[:8]}.csv"
    )

if __name__ == "__main__":
    create_sample_video()
    # Pre-warm deep model in memory
    init_shared_torch_model()
    print("==================================================")
    print(" ClipCluster AI Video Summarization & Clustering")
    print(" Server starting on http://127.0.0.1:5000")
    print("==================================================")
    app.run(host="127.0.0.1", port=5000, debug=True)
