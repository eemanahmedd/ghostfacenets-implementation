import os
import cv2
import sys
import time
import numpy as np
import pandas as pd
import tensorflow as tf
from flask import Flask, request, jsonify
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from sklearn.preprocessing import normalize
from mtcnn.mtcnn import MTCNN
from dotenv import load_dotenv

evaluator = None
known_faces = None

class EvalSingleImage:
    def __init__(self, model_interf):
        if isinstance(model_interf, str) and model_interf.endswith(".h5"):
            self.model = tf.keras.models.load_model(model_interf, compile=False)
            self.detector = MTCNN()
        else:
            self.model = model_interf
        self.dist_func = lambda a, b: np.dot(a, b)

    def prepare_image(self, image_path):
        img = load_img(image_path)
        img = img_to_array(img)
        img = self.crop_and_detect_face(img)
        if img is not None:
            img = (img - 127.5) * 0.0078125
            img = np.expand_dims(img, axis=0)
            return img
        else:
            return None

    def crop_and_detect_face(self, img):
        height, width, _ = img.shape
        crop_width_multiplier = 0.7
        crop_width = int(width * crop_width_multiplier)
        cropped_image = img[:, :crop_width, :]  # crop the left side of the image
        faces = self.detector.detect_faces(np.array(cropped_image))

        if len(faces) > 0:
            face = faces[0]
            x, y, width, height = face['box']
            face1 = img[y:y + height, x:x + width]
            resized_face = cv2.resize(face1, (112, 112))
            return resized_face
        else:
            return None

    def get_embeddings(self, img):
        emb = self.model.predict(img)
        emb = normalize(np.array(emb).astype("float32"))[0]
        return emb

    def compare_images(self, img_path1, img_path2):
        img1 = self.prepare_image(img_path1)
        img2 = self.prepare_image(img_path2)

        if img1 is not None or img2 is not None:
            emb1 = self.get_embeddings(img1)
            emb2 = self.get_embeddings(img2)
            return self.dist_func(emb1, emb2)
        else:
            return None

def get_known_embeddings(upload_folder, evaluator):
    known_faces = {}

    for people in os.listdir(upload_folder):
        people_dir = os.path.join(upload_folder, people)
        encoding_list = []

        for filename in os.listdir(people_dir):
            encoding_dict = {}
            image_path = os.path.join(people_dir, filename)
            face = evaluator.prepare_image(image_path)
            if face is not None:
                face_embedding = evaluator.get_embeddings(face)

                encoding_dict['filename'] = filename
                encoding_dict['embedding'] = face_embedding
                encoding_list.append(encoding_dict)
            else:
                print(f"No faces found in {filename}")

        known_faces[people] = encoding_list
    return known_faces

def convert_numpy_float32(obj):
    if isinstance(obj, np.float32):
        return float(obj)
    raise TypeError("Object of type {} is not JSON serializable".format(type(obj)))

# Load environment variables from .env file
load_dotenv()

# Access the variables
model_path = os.getenv("model_path")
upload_folder = os.getenv("upload_folder")

if evaluator is None:
    evaluator = EvalSingleImage(model_path)

if known_faces is None:
    known_faces = get_known_embeddings(upload_folder, evaluator)
    
app = Flask(__name__)

@app.route('/process_frame', methods=['POST'])
def process_frame():
    start_time = time.time()

    if request.method == 'POST':
        # Assuming the frame data is sent in the request
        data = request.files['image']
        data.save('img.jpg')
        face = evaluator.prepare_image('img.jpg')

        # Calculate embeddings using your face recognition module
        face_embedding = evaluator.get_embeddings(face)
        result_list = []

        # Compare with known embeddings
        for person, file_and_embs_list in known_faces.items():
            for file_dict in file_and_embs_list:
                filename = file_dict['filename']
                embedding = file_dict['embedding']
                similarity = evaluator.dist_func(embedding, face_embedding)

                temp_dict = {'person': person, 'filename': filename, 'similarity': float(similarity)}
                result_list.append(temp_dict)
            
        end_time = time.time()
        print(f'Time taken: {end_time - start_time}')
        return jsonify(result_list)

if __name__ == '__main__':
    size_of_function = sys.getsizeof(process_frame)
    print(f"Size of function: {size_of_function} bytes")
    app.run(debug=True)
