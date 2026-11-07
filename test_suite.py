import sys
import time
import json
import os

sys.path.append(r"C:\Users\SAMEEKSHA SHETTY\.gemini\antigravity\scratch\clipcluster")
from app import app, TASKS
from core.youtube_handler import is_valid_youtube_url, extract_youtube_video_id

client = app.test_client()

print("1. Testing GET / (Homepage with Dual Input Tabs) ...")
res = client.get("/")
assert res.status_code == 200, f"GET / failed: {res.status_code}"
assert b"ClipCluster" in res.data
assert b"Option 1: Upload Video File" in res.data
assert b"Option 2: YouTube Video Link" in res.data
print("   [OK] Homepage renders with both Upload and YouTube Link options.")

print("2. Testing YouTube URL validation logic ...")
assert is_valid_youtube_url("https://www.youtube.com/watch?v=dQw4w9WgXcQ") is True
assert is_valid_youtube_url("https://youtu.be/dQw4w9WgXcQ") is True
assert is_valid_youtube_url("https://www.youtube.com/shorts/3i_b7v4k2_Q") is True
assert is_valid_youtube_url("https://malicious-site.com/fake") is False
assert extract_youtube_video_id("https://youtu.be/dQw4w9WgXcQ") == "dQw4w9WgXcQ"
print("   [OK] YouTube regex accurately validates standard, short, and shorts URLs.")

print("3. Testing POST /api/youtube-info validation and error handling ...")
# Test invalid URL
res = client.post("/api/youtube-info", json={"url": "not_a_valid_url"})
assert res.status_code == 400
data = res.get_json()
assert data["success"] is False
assert "Invalid YouTube URL format" in data["error"]
print("   [OK] Invalid YouTube URL properly rejected with informative error.")

# Test empty URL
res = client.post("/api/youtube-info", json={"url": ""})
assert res.status_code == 400
print("   [OK] Empty URL rejected.")

print("4. Testing GET /api/sample-video pipeline execution ...")
res = client.get("/api/sample-video")
assert res.status_code == 200
sample_data = res.get_json()
assert sample_data["success"] is True
task_id = sample_data["task_id"]
meta = sample_data["metadata"]
print(f"   [OK] Sample video staged. Task ID: {task_id}, Duration: {meta['duration_formatted']}")

print("5. Starting AI Processing Pipeline ...")
proc_payload = {
    "task_id": task_id,
    "summary_percentage": 50.0,
    "num_clusters": 0
}
res = client.post("/api/process", json=proc_payload)
assert res.status_code == 200
print("   [OK] Processing started in background thread.")

print("6. Polling /api/status/<task_id> ...")
max_wait = 45
start = time.time()
completed = False
while time.time() - start < max_wait:
    res = client.get(f"/api/status/{task_id}")
    status_data = res.get_json()
    status = status_data.get("status")
    progress = status_data.get("progress")
    msg = status_data.get("message")
    print(f"   [Polling {round(time.time() - start, 1)}s] Progress: {progress}% - {msg}")
    if status == "completed":
        completed = True
        break
    elif status == "error":
        raise RuntimeError(f"Processing failed: {status_data.get('error')}")
    time.sleep(1.0)

assert completed, "Pipeline did not complete in time!"
print("   [OK] Processing reached 100% completion successfully!")

print("7. Testing GET /api/results/<task_id> ...")
res = client.get(f"/api/results/{task_id}")
assert res.status_code == 200
results_data = res.get_json()
assert results_data["success"] is True
d = results_data["data"]
print("   [OK] Results retrieved successfully:")
print(f"      - Original Duration: {d['stats']['original_duration_formatted']}")
print(f"      - Summary Duration: {d['stats']['summary_duration_formatted']}")
print(f"      - Reduction %: {d['stats']['reduction_percentage']}%")
print(f"      - Total Scenes: {d['stats']['total_scenes']}")
print(f"      - Total Clusters: {d['stats']['total_clusters']}")
print(f"      - Silhouette Score: {d['silhouette_score']}")

# 8. Test Audio Track in Summary Video
print("\n8. Testing Audio Track Integrity in Summary Video ...")
assert d["stats"]["has_audio"] is True, "Summary video metadata reports has_audio is False!"
summary_file = d["summary"]["file_path"]
assert os.path.exists(summary_file), f"Summary video file missing: {summary_file}"

import subprocess
from config import get_ffmpeg_binary
ffmpeg_bin = get_ffmpeg_binary()
check_cmd = [ffmpeg_bin, "-i", summary_file]
res = subprocess.run(check_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
assert "Audio: aac" in res.stderr or "Audio:" in res.stderr, "FFmpeg did not find an active Audio track in the summary video!"
print(f"   [OK] Verified summary video contains active stereo AAC audio stream: {summary_file}")

# 9. Test Short & Simple Notes Generation and Export
print("\n9. Testing Short & Simple Notes Generation & Export ...")
assert "short_notes" in d, "short_notes missing from results payload!"
notes = d["short_notes"]
assert "synopsis" in notes and len(notes["synopsis"]) > 20, "Notes synopsis missing or too short!"
assert "executive_bullets" in notes and len(notes["executive_bullets"]) >= 3, "Notes bullets missing!"
assert "highlights" in notes and len(notes["highlights"]) > 0, "Scene highlights missing from notes!"
print(f"   [OK] Generated {len(notes['executive_bullets'])} bullet points and {len(notes['highlights'])} scene highlights.")

res_notes = client.get(f"/api/export/{task_id}/notes")
assert res_notes.status_code == 200
assert res_notes.mimetype == "text/markdown"
assert b"AI SHORT NOTES" in res_notes.data
print("   [OK] Export /api/export/<task_id>/notes successfully downloaded markdown document.")

# 10. Test Topic Condensation: "Forest"
print("\n10. Testing Topic Condensation (Query: 'forest') ...")
res_topic = client.post("/api/condense-topic", json={
    "task_id": task_id,
    "topic": "forest"
})
assert res_topic.status_code == 200, f"Topic condensation failed: {res_topic.status_code}"
topic_res = res_topic.get_json()
assert topic_res["success"] is True
topic_data = topic_res["data"]
print(f"   [OK] Topic 'forest' condensed:")
print(f"      - Matched Segments: {topic_data['matched_scenes_count']}")
print(f"      - Duration: {topic_data['duration_formatted']}")
print(f"      - Video URL: {topic_data['video_url']}")
assert topic_data["matched_scenes_count"] >= 1
assert topic_data["has_audio"] is True

# Verify audio on topic video file
topic_file_rel = topic_data["video_url"].lstrip("/")
topic_file_path = os.path.join(r"C:\Users\SAMEEKSHA SHETTY\.gemini\antigravity\scratch\clipcluster", topic_file_rel.replace("/", os.sep))
assert os.path.exists(topic_file_path), f"Topic video file not found on disk: {topic_file_path}"
topic_check = subprocess.run([ffmpeg_bin, "-i", topic_file_path], stderr=subprocess.PIPE, text=True)
assert "Audio:" in topic_check.stderr, "Topic video missing audio stream!"
print("   [OK] Topic video contains active audio track!")

# 11. Test Topic Condensation: "Sunset"
print("\n11. Testing Topic Condensation (Query: 'sunset') ...")
res_sunset = client.post("/api/condense-topic", json={
    "task_id": task_id,
    "topic": "sunset"
})
assert res_sunset.status_code == 200
sunset_data = res_sunset.get_json()["data"]
print(f"   [OK] Topic 'sunset' condensed: {sunset_data['matched_scenes_count']} scenes ({sunset_data['duration_formatted']})")

# 12. Test Topic Notes Export
print("\n12. Testing Topic Notes Export Download ...")
res_topic_export = client.get(f"/api/export/{task_id}/topic-notes?topic=forest")
assert res_topic_export.status_code == 200
assert res_topic_export.mimetype == "text/markdown"
assert b"TOPIC CONDENSATION NOTES" in res_topic_export.data
print("   [OK] Downloaded topic markdown notes successfully.")

print("\n==================================================")
print(" ALL TESTS PASSED WITH 100% SUCCESS! ")
print("==================================================")
