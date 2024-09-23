import cv2
import numpy as np
from keras.models import load_model
from statistics import mode
from utils.datasets import get_labels
from utils.inference import (
    draw_text,
    draw_bounding_box,
    apply_offsets,
    load_detection_model,
)


# Replace scipy.misc functions with OpenCV equivalents
def _imread(image_name):
    return cv2.imread(image_name)


def _imresize(image_array, size):
    return cv2.resize(image_array, size)


# Convert integer classes to categorical
def to_categorical(integer_classes, num_classes=2):
    integer_classes = np.asarray(integer_classes, dtype="int")
    num_samples = integer_classes.shape[0]
    categorical = np.zeros((num_samples, num_classes))
    categorical[np.arange(num_samples), integer_classes] = 1
    return categorical


# Preprocess input image
def preprocess_input(x, v2=True):
    x = x.astype("float32")
    x = x / 255.0
    if v2:
        x = x - 0.5
        x = x * 2.0
    return x


# Parameters and models
USE_WEBCAM = True
emotion_model_path = "./models/emotion_model.hdf5"
emotion_labels = get_labels("fer2013")
frame_window = 10
emotion_offsets = (20, 40)

# Load models
face_cascade = cv2.CascadeClassifier("./models/haarcascade_frontalface_default.xml")
emotion_classifier = load_model(emotion_model_path)
emotion_target_size = emotion_classifier.input_shape[1:3]
emotion_window = []

# Video capture setup
cv2.namedWindow("window_frame")
cap = cv2.VideoCapture(0) if USE_WEBCAM else cv2.VideoCapture("./demo/dinner.mp4")

# Main loop
while cap.isOpened():
    ret, bgr_image = cap.read()
    if not ret:
        break

    gray_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY)
    rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)

    faces = face_cascade.detectMultiScale(
        gray_image,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
        flags=cv2.CASCADE_SCALE_IMAGE,
    )

    for face_coordinates in faces:
        x1, x2, y1, y2 = apply_offsets(face_coordinates, emotion_offsets)
        gray_face = gray_image[y1:y2, x1:x2]
        try:
            gray_face = cv2.resize(gray_face, (emotion_target_size))
        except Exception as e:
            print("Exception during face resizing:", e)
            continue

        gray_face = preprocess_input(gray_face, True)
        gray_face = np.expand_dims(gray_face, 0)
        gray_face = np.expand_dims(gray_face, -1)

        emotion_prediction = emotion_classifier.predict(gray_face)
        emotion_probability = np.max(emotion_prediction)
        emotion_label_arg = np.argmax(emotion_prediction)
        emotion_text = emotion_labels[emotion_label_arg]
        emotion_window.append(emotion_text)

        if len(emotion_window) > frame_window:
            emotion_window.pop(0)
        try:
            emotion_mode = mode(emotion_window)
        except Exception as e:
            print("Exception during mode calculation:", e)
            continue

        color = None
        if emotion_text == "angry":
            color = emotion_probability * np.asarray((255, 0, 0))
        elif emotion_text == "sad":
            color = emotion_probability * np.asarray((0, 0, 255))
        elif emotion_text == "happy":
            color = emotion_probability * np.asarray((255, 255, 0))
        elif emotion_text == "surprise":
            color = emotion_probability * np.asarray((0, 255, 255))
        else:
            color = emotion_probability * np.asarray((0, 255, 0))

        color = color.astype(int)
        color = color.tolist()

        draw_bounding_box(face_coordinates, rgb_image, color)
        draw_text(face_coordinates, rgb_image, emotion_mode, color, 0, -45, 1, 1)

    bgr_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
    cv2.imshow("window_frame", bgr_image)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
