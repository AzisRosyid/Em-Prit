import os
import cv2
import numpy as np
import tensorflow as tf
from ultralytics import YOLO
from flask import Flask, request, render_template, redirect, url_for
from werkzeug.utils import secure_filename
import base64
import json

app = Flask(__name__, template_folder="tf_templates")
UPLOAD_FOLDER = "uploads"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

print("Memuat model-model AI...")

# 1. Muat Model Deteksi Burung (YOLOv8 - PyTorch)
try:
    yolo_model = YOLO("yolov8n.pt")
    print("Model YOLOv8 berhasil dimuat.")
except Exception as e:
    print(f"Error memuat model YOLO: {e}")
    yolo_model = None

# 2. Muat Model Klasifikasi Spesies (TensorFlow/Keras)
CLASSIFIER_MODEL_PATH = "bird_classifier.keras"
if not os.path.exists(CLASSIFIER_MODEL_PATH):
    CLASSIFIER_MODEL_PATH = "bird_classifier.h5"

try:
    classifier_model = tf.keras.models.load_model(CLASSIFIER_MODEL_PATH)
    print(f"Model Klasifikasi '{CLASSIFIER_MODEL_PATH}' berhasil dimuat.")
    try:
        train_dir = "bird_species_dataset/data/Train"
        if not os.path.exists(train_dir):
            train_dir = "archive/train"

        class_names = sorted(os.listdir(train_dir))
        with open("class_names.json", "w") as f:
            json.dump(class_names, f)
        print(f"Nama kelas berhasil disimpan ke class_names.json")
    except FileNotFoundError:
        print(
            "Warning: Tidak dapat menemukan direktori training untuk memuat nama kelas. Pastikan path benar."
        )
        class_names = [f"Class {i}" for i in range(220)]

except Exception as e:
    print(f"Error memuat model klasifikasi: {e}")
    print("Pastikan file model ada dan Anda sudah menjalankan train_classifier.py")
    classifier_model = None
    class_names = []


def analyze_image(image_path):
    """
    Menjalankan pipeline AI dua tahap pada satu gambar.
    """
    if not yolo_model or not classifier_model:
        return {"error": "Satu atau lebih model AI gagal dimuat."}

    original_image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)

    # --- Tahap 1: Deteksi Burung dengan YOLO ---
    yolo_results = yolo_model(image_rgb, verbose=False)
    annotated_image = original_image.copy()

    analysis_results = []

    for box in yolo_results[0].boxes:
        class_id = int(box.cls[0])
        if yolo_model.names[class_id] == "bird":
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            # --- Tahap 2: Klasifikasi Spesies dengan TensorFlow ---

            cropped_bird = original_image[y1:y2, x1:x2]

            if cropped_bird.size > 0:
                img_resized = cv2.resize(cropped_bird, (224, 224))
                img_array = tf.keras.preprocessing.image.img_to_array(img_resized)
                img_array = tf.expand_dims(img_array, 0)
                img_array = img_array / 255.0

                predictions = classifier_model.predict(img_array, verbose=0)
                score = tf.nn.softmax(predictions[0])

                predicted_class = class_names[np.argmax(score)]
                confidence = 100 * np.max(score)

                is_pest = any(
                    pest in predicted_class.lower() for pest in ["sparrow", "pipit"]
                )
                action = "AKTIFKAN PENGUSIR" if is_pest else "ABAIKAN (Non-Hama)"

                color = (0, 0, 255) if is_pest else (0, 255, 0)
                cv2.rectangle(annotated_image, (x1, y1), (x2, y2), color, 2)
                label = f"{predicted_class}: {confidence:.1f}%"
                cv2.putText(
                    annotated_image,
                    label,
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    color,
                    2,
                )

                _, buffer = cv2.imencode(".jpg", cropped_bird)
                cropped_b64 = base64.b64encode(buffer).decode("utf-8")

                analysis_results.append(
                    {
                        "species": predicted_class,
                        "confidence": f"{confidence:.2f}",
                        "action": action,
                        "cropped_image": cropped_b64,
                    }
                )

    _, buffer = cv2.imencode(".jpg", annotated_image)
    annotated_image_b64 = base64.b64encode(buffer).decode("utf-8")

    return {"annotated_image": annotated_image_b64, "detections": analysis_results}


@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        if "file" not in request.files:
            return redirect(request.url)
        file = request.files["file"]
        if file.filename == "":
            return redirect(request.url)

        if file:
            filename = secure_filename(file.filename)
            image_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
            file.save(image_path)

            results = analyze_image(image_path)

            with open(image_path, "rb") as f:
                original_img_b64 = base64.b64encode(f.read()).decode("utf-8")

            return render_template(
                "result.html",
                filename=filename,
                results=results,
                original_image=original_img_b64,
            )

    return render_template("index.html")


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
