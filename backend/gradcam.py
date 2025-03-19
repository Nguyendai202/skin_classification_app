import tensorflow as tf
import numpy as np
import cv2
import os
from datetime import datetime
import base64
class GradCAM:
    def __init__(self, model, output_dir="gradcam_results"):
        self.model = model
        self.output_dir = output_dir
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

    def get_gradcam(self, img_array, layer_name, class_index):
        """Tính Grad-CAM từ mô hình và ảnh đầu vào"""
        grad_model = tf.keras.models.Model(
            [self.model.model.inputs], 
            [self.model.model.get_layer(layer_name).output, self.model.model.output]
        )

        with tf.GradientTape() as tape:
            conv_outputs, predictions = grad_model(img_array)
            loss = predictions[:, class_index]

        grads = tape.gradient(loss, conv_outputs)[0]
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

        conv_outputs = conv_outputs[0]
        heatmap = tf.reduce_mean(tf.multiply(conv_outputs, pooled_grads), axis=-1)
        heatmap = np.maximum(heatmap, 0)
        heatmap /= np.max(heatmap) if np.max(heatmap) != 0 else 1

        return heatmap

    def overlay_gradcam(self, img, heatmap, alpha=0.4):
        """Chồng heatmap Grad-CAM lên ảnh gốc"""
        heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
        heatmap = np.uint8(255 * heatmap)
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

        superimposed_img = heatmap * alpha + img
        superimposed_img = np.clip(superimposed_img, 0, 255).astype(np.uint8)
        
        return superimposed_img

    def generate_gradcam(self, image, filename=None, layer_name='conv5_block32_concat'):
        """Generate Grad-CAM visualization for an image"""
        try:
            # Preprocess image
            img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            img_processed = tf.keras.applications.densenet.preprocess_input(img_rgb)
            img_array = np.expand_dims(img_processed, axis=0)

            # Get prediction with probabilities
            result = self.model.image_classify(image)
            label_index = result['class_id']
            probabilities = result['probabilities']

            # Generate Grad-CAM
            heatmap = self.get_gradcam(img_array, layer_name, label_index)
            gradcam_img = self.overlay_gradcam(image, heatmap)
            gradcam_rgb = cv2.cvtColor(gradcam_img, cv2.COLOR_BGR2RGB)

            # Save results
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            if filename:
                base_filename = os.path.splitext(filename)[0]
                save_filename = f"{base_filename}_gradcam_{timestamp}.jpg"
            else:
                save_filename = f"gradcam_{timestamp}.jpg"

            save_path = os.path.join(self.output_dir, save_filename)
            cv2.imwrite(save_path, gradcam_rgb)

            # Create base64 representation
            _, buffer = cv2.imencode('.jpg', gradcam_rgb)
            image_base64 = base64.b64encode(buffer).decode('utf-8')

            return {
                "label_index": label_index,
                "probabilities": probabilities,
                "file_path": save_path,
                "gradcam_image": image_base64
            }

        except Exception as e:
            print(f"Error generating Grad-CAM: {str(e)}")
            return None