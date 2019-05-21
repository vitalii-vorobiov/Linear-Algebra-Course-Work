from flask import Flask, request, jsonify
import base64
import cv2
import os
import numpy as np
import math
from PIL import Image
import io

train_folder = "../dataset/train-data"
test_folder = "../dataset/test-data"
FACE_CASCADE = "../haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(FACE_CASCADE)


def add_cors_headers(response):
    response.headers['Access-Control-Allow-Origin'] = '*'
    if request.method == 'OPTIONS':
        response.headers['Access-Control-Allow-Methods'] = 'DELETE, GET, POST, PUT'
        headers = request.headers.get('Access-Control-Request-Headers')
        if headers:
            response.headers['Access-Control-Allow-Headers'] = headers
    return response


def string_to_image(base64_string):
    img_data = base64.b64decode(base64_string)
    return Image.open(io.BytesIO(img_data))


def cropp_image(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    faces = faceCascade.detectMultiScale(
        image,
        scaleFactor=1.3,
        minNeighbors=5,
        minSize=(10, 10),
        flags=cv2.CASCADE_SCALE_IMAGE
    )

    for (x, y, w, h) in faces:
        image = image[y:y + h, x:x + w]

    image = cv2.resize(image, (100, 100))

    return image


def vector_to_image(vector, width, height):
    mat = []
    for i in range(height):
        mat.append([])
        for j in range(width):
            mat[i].append(vector[i * width + j])

    return mat


class Database:
    def __init__(self, folder, img_width, img_height):
        self.folder = folder
        self.img_width = img_width
        self.img_height = img_height
        self.people_images, self.all_images = self.open_images()
        self.global_mean_face = self.calculate_mean_face()

    def open_images(self):
        people_images = {}
        for folder_name in os.listdir(self.folder):
            people_images[folder_name] = []
            for image_name in os.listdir(os.path.join(self.folder, folder_name)):
                image = cv2.imread(os.path.join(self.folder, folder_name, image_name), 0).astype(np.float32)
                people_images[folder_name].append(image.flatten())

        width = sum(len(value) for key, value in people_images.items())
        all_images = np.ndarray(shape=(self.img_width * self.img_height, width))

        counter = 0
        for key, value in people_images.items():
            for image in value:
                all_images[:, counter] = image
                counter += 1

        return people_images, all_images

    def calculate_mean_face(self):
        mean_face = np.ndarray(shape=(self.img_height * self.img_width))

        for i in range(len(self.all_images[0])):
            mean_face = mean_face + self.all_images[:, i]

        return mean_face / len(self.all_images[0])

    def normalize_images(self):
        images = np.array(self.all_images, copy=True)

        for i in range(images.shape[1]): # 360
            images[:, i] -= self.global_mean_face

        return images


db = Database(train_folder, 100, 100)

A_matrix = db.normalize_images()
cov_matrix = np.cov(A_matrix.transpose())

e_values, e_vectors = np.linalg.eig(cov_matrix)

# Our final 360 eigenfaces | matrix dimension (10304, 360)
features = np.matmul(e_vectors, A_matrix.transpose()).transpose()

app = Flask(__name__)
app.after_request(add_cors_headers)


@app.route("/")
def home():
    return "Ok", 200


@app.route("/recognize", methods=["POST"])
def recognize():
    data = request.get_json()[23:]
    image = cropp_image(np.array(string_to_image(data))).flatten()

    image_normalized = image - db.global_mean_face

    person_weights = []
    for i in range(features.shape[1]):
        weight = np.dot(image_normalized, features[:, i])
        person_weights.append(weight)

    person_weights = np.array(person_weights)

    db_weights = []

    for i in range(A_matrix.shape[1]):
        db_weights.append([])
        for j in range(features.shape[1]):
            weight = np.dot(A_matrix[:, i], features[:, j])
            db_weights[i].append(weight)

    db_weights = np.array(db_weights)

    res = []

    for weight in db_weights:
        res.append(np.linalg.norm(weight - person_weights))

    return jsonify(list(db.people_images.keys())[math.floor((res.index(min(res)) + 1) / 10)]), 200


if __name__ == '__main__':
    app.run()
