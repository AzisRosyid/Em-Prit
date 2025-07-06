# app.py
# Main application for the Em-Prit Bird Detection System.
# This version provides a choice between live webcam detection and video file analysis.
#
# To run:
# 1. Create a folder named 'static' and another named 'templates'.
# 2. Place this file in the root folder. Place all .html files in 'templates'.
# 3. Install libraries: pip install -r requirements.txt
# 4. Run the script: python app.py
# 5. Open your web browser and go to http://127.0.0.1:5000

import os
import cv2
from flask import Flask, render_template, request, redirect, url_for, Response
from ultralytics import YOLO
from werkzeug.utils import secure_filename
import math

app = Flask(__name__)

# --- Configuration ---
UPLOAD_FOLDER = "static"
ALLOWED_EXTENSIONS = {"mp4", "mov", "avi", "mkv"}
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# --- AI Model Configuration ---
# Load the model once to be used by both live feed and analysis
try:
    model = YOLO("yolov8n.pt")
    print("YOLOv8n model loaded successfully.")
except Exception as e:
    print(f"Error loading YOLO model: {e}")
    exit()

TARGET_CLASS_ID = 14
TARGET_CLASS_NAME = "bird"
CONFIDENCE_THRESHOLD = 0.4


def allowed_file(filename):
    """Checks if the uploaded file has an allowed extension."""
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def format_time(seconds):
    """Converts seconds to a MM:SS format."""
    minutes = math.floor(seconds / 60)
    remaining_seconds = int(seconds % 60)
    return f"{minutes:02d}:{remaining_seconds:02d}"


# --- Routes ---


@app.route("/")
def index():
    """Main page with choices for live feed or video upload."""
    return render_template("index.html")


@app.route("/live_page")
def live_page():
    """Page to display the live webcam feed."""
    return render_template("live.html")


@app.route("/upload_page")
def upload_page():
    """Page to upload a video file for analysis."""
    return render_template("upload.html")


def generate_live_frames():
    """
    Generator function for live webcam feed.
    Captures video, runs YOLOv8 inference, and yields frames.
    """
    camera = cv2.VideoCapture(0)
    if not camera.isOpened():
        print("Error: Could not open webcam.")
        return

    while True:
        success, frame = camera.read()
        if not success:
            break

        results = model(frame, stream=True, verbose=False)
        is_bird_detected = False

        for r in results:
            boxes = r.boxes
            for box in boxes:
                cls_id = int(box.cls[0])
                if cls_id == TARGET_CLASS_ID:
                    confidence = float(box.conf[0])
                    if confidence > CONFIDENCE_THRESHOLD:
                        is_bird_detected = True
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                        label = f"{TARGET_CLASS_NAME}: {confidence:.2f}"
                        cv2.putText(
                            frame,
                            label,
                            (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.6,
                            (0, 0, 255),
                            2,
                        )

        status_text = "BIRD DETECTED!" if is_bird_detected else "CLEAR"
        status_color = (0, 0, 255) if is_bird_detected else (0, 255, 0)
        cv2.putText(
            frame,
            f"Status: {status_text}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            status_color,
            2,
        )

        ret, buffer = cv2.imencode(".jpg", frame)
        if not ret:
            continue
        frame_bytes = buffer.tobytes()
        yield (
            b"--frame\r\n" b"Content-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n"
        )

    camera.release()


@app.route("/video_feed")
def video_feed():
    """Video streaming route for the live feed."""
    return Response(
        generate_live_frames(), mimetype="multipart/x-mixed-replace; boundary=frame"
    )


@app.route("/analyze", methods=["POST"])
def analyze_video():
    """Handles video upload and triggers the analysis."""
    if "video" not in request.files:
        return redirect(url_for("upload_page"))
    file = request.files["video"]
    if file.filename == "":
        return redirect(url_for("upload_page"))

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        video_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        file.save(video_path)

        camera = cv2.VideoCapture(video_path)
        fps = camera.get(cv2.CAP_PROP_FPS)
        timestamps = []
        frame_count = 0

        print(f"Analyzing video: {filename} at {fps:.2f} FPS")

        while True:
            success, frame = camera.read()
            if not success:
                break

            frame_count += 1
            results = model(frame, stream=True, verbose=False)

            is_bird_detected_in_frame = False
            for r in results:
                boxes = r.boxes
                for box in boxes:
                    if (
                        int(box.cls[0]) == TARGET_CLASS_ID
                        and float(box.conf[0]) > CONFIDENCE_THRESHOLD
                    ):
                        is_bird_detected_in_frame = True
                        break
                if is_bird_detected_in_frame:
                    break

            if is_bird_detected_in_frame:
                current_time_sec = frame_count / fps
                if not timestamps or (
                    current_time_sec - timestamps[-1]["seconds"] > 1.0
                ):
                    timestamps.append(
                        {
                            "seconds": current_time_sec,
                            "formatted": format_time(current_time_sec),
                        }
                    )

        camera.release()
        print(f"Analysis complete. Found {len(timestamps)} detection events.")

        return render_template(
            "results.html", video_filename=filename, timestamps=timestamps
        )

    return redirect(url_for("upload_page"))


if __name__ == "__main__":
    if not os.path.exists(UPLOAD_FOLDER):
        os.makedirs(UPLOAD_FOLDER)
    app.run(debug=True)
