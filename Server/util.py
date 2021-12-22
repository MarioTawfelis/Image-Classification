import base64
import json

import cv2 as cv
import joblib
import numpy as np
from wavelet import w2d

# Private global variables
__class_name_to_number = {}
__class_number_to_name = {}
__model = None


def classify_image(image_b64, file_path=None):
    imgs = get_cropped_image_if_valid(file_path, image_b64)

    result = []

    for img in imgs:
        # 1 - Rescale raw image to 32x32
        scaled_raw_img = cv.resize(img, (32, 32))

        # 2 - Create a WT image
        img_har = w2d(img, 'db1', 5)

        # 3 - Rescale the transformed image to 32x32
        scaled_img_har = cv.resize(img_har, (32, 32))

        # 4 - Vertically stack the raw and transformed images
        stacked_img = np.vstack((scaled_raw_img.reshape(32 * 32 * 3, 1), scaled_img_har.reshape(32 * 32, 1)))

        # 5 - Reshape and convert to float
        len_image_array = (32 * 32 * 3) + (32 * 32)
        X = stacked_img.reshape(1, len_image_array).astype(float)

        # 6 - Predict
        result.append(class_number_to_name(__model.predict(X)[0]))

    return result


def get_cv2_image_from_base64_string(b64str):
    """
    credit: https://stackoverflow.com/questions/33754935/read-a-base-64-encoded-image-from-memory-using-opencv-python-library
    """
    encoded_data = b64str.split(',')[1]
    nparr = np.frombuffer(base64.b64decode(encoded_data), np.uint8)
    img = cv.imdecode(nparr, cv.IMREAD_COLOR)
    return img


def load_saved_artifacts():
    print("Loading saved artifacts...")
    global __class_name_to_number
    global __class_number_to_name

    with open("./artifacts/class_dictionary.json", "r") as f:
        __class_name_to_number = json.load(f)
        __class_number_to_name = {v: k for k, v in __class_name_to_number.items()}

    global __model
    if __model is None:
        with open('./artifacts/saved_model.pkl', 'rb') as f:
            __model = joblib.load(f)
    print("Loading saved artifacts...Done!")


def get_cropped_image_if_valid(image_path, image_b64):
    # Load pretrained models to detect face and eyes
    face_cascade = cv.CascadeClassifier('./opencv/haarcascades/haarcascade_frontalface_default.xml')
    eye_cascade = cv.CascadeClassifier('./opencv/haarcascades/haarcascade_eye.xml')

    # Read image
    if image_path:
        img = cv.imread(image_path)
    else:
        img = get_cv2_image_from_base64_string(image_b64)

    # Convert image to grayscale
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    # Detect face
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    cropped_faces = []
    # Iterate over all faces
    for (x, y, w, h) in faces:
        # ROI: Region Of Interest
        roi_gray = gray[y:y + h, x:x + w]
        roi_color = img[y:y + h, x:x + w]

        # Detect eyes
        eyes = eye_cascade.detectMultiScale(roi_gray)
        if len(eyes) >= 2:
            cropped_faces.append(roi_color)

    return cropped_faces


def get_b64_test_image():
    with open("b64.txt") as f:
        return f.read()


def class_number_to_name(class_num):
    return __class_number_to_name[class_num]


if __name__ == "__main__":
    load_saved_artifacts()
    print(classify_image(get_b64_test_image(), None))
