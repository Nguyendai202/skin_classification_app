from fastapi import FastAPI, UploadFile, File
from predictions.classifications import ClassificationModel
import zipfile
import os
import io
import tempfile
from typing import List
from definitions import model_path
import uvicorn
import numpy as np
import cv2
import time
from starlette.responses import RedirectResponse
import concurrent.futures
import logging
import base64
import tensorflow as tf
from fastapi.middleware.cors import CORSMiddleware
from datetime import datetime
import os
from gradcam import GradCAM


# Thiết lập logging
logging.basicConfig(filename='error.log', level=logging.ERROR,
                    format='%(asctime)s|%(name)s|%(levelname)s|%(message)s')
RESULTS_DIR = "gradcam_results"
if not os.path.exists(RESULTS_DIR):
    os.makedirs(RESULTS_DIR)
app_desc = """<h2>HUS-VNU</h2>"""
app = FastAPI(title="Nguyễn Đài", description=app_desc)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Khởi tạo mô hình
predict = ClassificationModel(model_path)

# Initialize GradCAM with the model
gradcam_generator = GradCAM(predict)

# Danh sách nhãn
unique_dx = ['bkl', 'nv', 'df', 'mel', 'vasc', 'bcc', 'akiec']

@app.get("/", include_in_schema=False)
async def index():
    return RedirectResponse(url="/docs")

# Update in the API response section to include labeled probabilities:
@app.post("/upload-images")
async def upload_images(files: List[UploadFile] = File(...)):
    results = {}
    times = []
    count_file = 0
    start_time_1 = time.time()

    try:
        for file in files:
            file_extension = os.path.splitext(file.filename)[1]
            if file_extension == '.zip':
                with tempfile.TemporaryDirectory() as temp_dir:
                    # ... existing zip handling code ...
                    for root, _, files in os.walk(temp_dir):
                        for filename in files:
                            count_file += 1
                            start_time = time.time()
                            image_path = os.path.join(root, filename)
                            image_array = cv2.imread(image_path)
                            with concurrent.futures.ThreadPoolExecutor() as executor:
                                future = executor.submit(perform_prediction_with_gradcam, image_array, filename)
                                result = future.result()
                                if result is not None:
                                    label_index, gradcam_data = result
                                    label = unique_dx[label_index]
                                    # Map probabilities to class names
                                    class_probabilities = {
                                        unique_dx[i]: prob 
                                        for i, prob in gradcam_data["probabilities"].items()
                                    }
                                    results[filename] = {
                                        "label": label,
                                        "probabilities": class_probabilities,
                                        "gradcam_path": gradcam_data["file_path"],
                                        "gradcam_image": gradcam_data["base64"]
                                    }
                            elapsed_time = time.time() - start_time
                            times.append(elapsed_time)
            else:
                # Xử lý ảnh đơn lẻ
                image_bytes = await file.read()
                image_array = np.frombuffer(image_bytes, dtype=np.uint8)
                img_bgr = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
                result = perform_prediction_with_gradcam(img_bgr, file.filename)
                if result is not None:
                    label_index, gradcam_data = result
                    label = unique_dx[label_index]
                    # Map probabilities to class names
                    class_probabilities = {
                        unique_dx[i]: prob 
                        for i, prob in gradcam_data["probabilities"].items()
                    }
                    results["image"] = {
                        "label": label,
                        "probabilities": class_probabilities,
                        "gradcam_path": gradcam_data["file_path"],
                        "gradcam_image": gradcam_data["base64"]
                    }
                elapsed_time = time.time() - start_time_1
                count_file += 1
                times.append(elapsed_time)

    except Exception as e:
        logging.error(f'Error uploading images from api: {str(e)}')

    return {
        "results": results,
        "times": times,
        "count_file": count_file,
    }

def get_gradcam(model, img_array, layer_name, class_index):
    """Tính Grad-CAM từ mô hình và ảnh đầu vào"""
    # Tạo mô hình con để lấy đầu ra từ lớp cuối cùng của convolution và dự đoán
    grad_model = tf.keras.models.Model(
        [model.model.inputs], [model.model.get_layer(layer_name).output, model.model.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        loss = predictions[:, class_index]

    # Tính gradient của loss đối với đầu ra của lớp convolution
    grads = tape.gradient(loss, conv_outputs)[0]
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # Nhân đầu ra convolution với gradient đã được pool
    conv_outputs = conv_outputs[0]
    heatmap = tf.reduce_mean(tf.multiply(conv_outputs, pooled_grads), axis=-1)
    heatmap = np.maximum(heatmap, 0)  # ReLU
    heatmap /= np.max(heatmap) if np.max(heatmap) != 0 else 1  # Chuẩn hóa

    return heatmap

def overlay_gradcam(img, heatmap, alpha=0.4):
    """Chồng heatmap Grad-CAM lên ảnh gốc"""
    # Resize heatmap về kích thước ảnh gốc
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)  # Áp dụng colormap

    # Chồng heatmap lên ảnh gốc
    superimposed_img = heatmap * alpha + img
    superimposed_img = np.clip(superimposed_img, 0, 255).astype(np.uint8)
    
    return superimposed_img

def perform_prediction_with_gradcam(image, filename=None):
    """Dự đoán nhãn và trả về ảnh với Grad-CAM"""
    try:
        result = gradcam_generator.generate_gradcam(image, filename)
        if result:
            return result["label_index"], {
                "base64": result["gradcam_image"],
                "file_path": result["file_path"],
                "probabilities": result["probabilities"]
            }
        return None
    except Exception as e:
        logging.error(f'Error performing Grad-CAM prediction: {str(e)}')
        return None



if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=6879, reload=True)