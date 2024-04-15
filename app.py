from flask import Flask, request, jsonify, send_from_directory
from werkzeug.utils import secure_filename
import os
import cv2
import numpy as np
from datetime import datetime
from tempfile import NamedTemporaryFile
import asyncio
import uuid

app = Flask(__name__)

# Function to check if an image is blurry
def is_blurry(image, threshold=200):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    variance = cv2.Laplacian(gray, cv2.CV_64F).var()
    return variance < threshold

# Function to extract frames from a video
async def extract_frames(video_path, frames_per_second, blur_threshold=20):
    try:
        frames = []
        if not os.path.exists(video_path):
            return "Video file not found.", 404
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        count = 0
        while True:
            ret, image = cap.read()
            if not ret:
                break
            if count % int(fps / frames_per_second) == 0:
                if not is_blurry(image, blur_threshold):
                    frames.append(image)
            count += 1
        cap.release()
        return frames
    except Exception as e:
        return f"Error processing video: {str(e)}", 500

# Function to load image paths from a folder
async def load_image_paths_from_folder(folder):
    image_paths = [os.path.join(folder, f) for f in os.listdir(folder) if f.endswith('.jpg')]
    return image_paths

# Function to evaluate resolution of an image
async def evaluate_resolution(image):
    if image is not None:
        return image.shape[0] * image.shape[1]
    return 0

# Function to evaluate sharpness of an image
async def evaluate_sharpness(image):
    if image is not None:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return cv2.Laplacian(gray, cv2.CV_64F).var()
    return 0

# Function to evaluate aesthetic score of an image
async def evaluate_aesthetic(image):
    if image is not None:
        b, g, r = cv2.split(image)
        b_var = np.var(b)
        g_var = np.var(g)
        r_var = np.var(r)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gabor_features = []
        for theta in range(4):
            theta = theta / 4. * np.pi
            kernel = cv2.getGaborKernel((21, 21), 5.0, theta, 10.0, 1, 0, ktype=cv2.CV_32F)
            filtered = cv2.filter2D(gray, cv2.CV_8UC3, kernel)
            gabor_features.extend([np.mean(filtered), np.var(filtered)])
        aesthetic_score = (b_var + g_var + r_var) + sum(gabor_features)
        return aesthetic_score
    return 0

# Function to calculate quality score of an image
async def quality_score(image):
    resolution_score = await evaluate_resolution(image)
    sharpness_score = await evaluate_sharpness(image)
    aesthetic_score = await evaluate_aesthetic(image)
    combined_score = resolution_score + sharpness_score + aesthetic_score
    return combined_score

# Function to select the best quality images
async def select_best_quality(frames, num_clusters):
    quality_scores = await asyncio.gather(*[quality_score(frame) for frame in frames])
    sorted_indexes = np.argsort(quality_scores)[::-1][:num_clusters]
    best_frames = [frames[i] for i in sorted_indexes]
    return best_frames

# Function to save best images to the static folder
def save_best_images(best_frames):
    os.makedirs('static', exist_ok=True)
    for i, frame in enumerate(best_frames):
        output_path = os.path.join('static', f"best_image_{i}.jpg")
        cv2.imwrite(output_path, frame)

# Route for the index page
@app.route("/", methods=["GET"])
def index():
    return "Welcome to the image processing service!"

# Route to process the uploaded video
@app.route("/process_video/", methods=["POST"])
async def process_video():
    try:
        frames_per_second = int(request.form["frames_per_second"])
        num_clusters = int(request.form["num_clusters"])
        if 'video_file' not in request.files:
            return jsonify({"error": "No video file provided."}), 400
        video_file = request.files["video_file"]
        if video_file.filename == '':
            return jsonify({"error": "No selected video file."}), 400
        with NamedTemporaryFile(delete=False) as temp_video_file:
            temp_video_file_path = temp_video_file.name
            video_file.save(temp_video_file_path)
            frames = await extract_frames(temp_video_file_path, frames_per_second)
            best_frames = await select_best_quality(frames, num_clusters)
            save_best_images(best_frames)
        os.remove(temp_video_file_path)
        return jsonify({"message": "Processing completed successfully."}), 200
    except Exception as e:
        return jsonify({"error": f"Internal Server Error: {str(e)}"}), 500

# Route to serve static image files
@app.route('/static/<path:image_filename>', methods=["GET"])
def serve_image(image_filename):
    try:
        image_path = os.path.join('static', image_filename)
        if not os.path.exists(image_path):
            return jsonify({"error": "Image not found."}), 404

        return send_from_directory('static', image_filename)
    except Exception as e:
        return jsonify({"error": f"Internal Server Error: {str(e)}"}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True)
