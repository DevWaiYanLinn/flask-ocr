import numpy as np
import cv2
from sklearn.preprocessing import LabelEncoder
import imutils
from imutils.contours import sort_contours
from keras.models import load_model

model = load_model("./Thuta.h5")


def test_pipeline(image_path):
    img = cv2.imread(image_path)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_gray = cv2.resize(img_gray, (1200, 318))
    img_gray = cv2.GaussianBlur(img_gray, (5, 5), 0)
    edged = cv2.Canny(img_gray, 30, 150)
    dilated = cv2.dilate(edged.copy(), None, iterations=6)
    normalized_image = cv2.normalize(dilated, None, 0, 255, cv2.NORM_MINMAX)

    contours = cv2.findContours(
        normalized_image.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    contours = imutils.grab_contours(contours)
    contours = sort_contours(contours, method="left-to-right")[0]
    labels = [
        "0",
        "1",
        "2",
        "3",
        "4",
        "5",
        "6",
        "7",
        "8",
        "9",
        "_",
        "-",
        "[",
        "]",
        "+",
        "%",
    ]

    real_labels = [
        "0",
        "1",
        "2",
        "3",
        "4",
        "5",
        "6",
        "7",
        "8",
        "9",
        "*",
        "-",
        "(",
        ")",
        "+",
        "/",
    ]

    label_encoder = LabelEncoder()
    label_class = label_encoder.fit_transform(labels)
     
    results = []

    for c in contours:
        if cv2.contourArea(c) < 1000:
            continue
        (x, y, w, h) = cv2.boundingRect(c)
        if 20 <= w:
            roi = img_gray[y : y + h, x : x + w]
            thresh = cv2.threshold(
                roi, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU
            )[1]
            (th, tw) = thresh.shape
            if tw > th:
                thresh = imutils.resize(thresh, width=28)
            if th > tw:
                thresh = imutils.resize(thresh, height=28)
            (th, tw) = thresh.shape
            dx = int(max(0, 28 - tw) / 2.0)
            dy = int(max(0, 28 - th) / 2.0)
            padded = cv2.copyMakeBorder(
                thresh,
                top=dy,
                bottom=dy,
                left=dx,
                right=dx,
                borderType=cv2.BORDER_CONSTANT,
                value=(0, 0, 0),
            )
            padded = cv2.resize(padded, (28, 28))
            padded = np.array(padded)
            padded = padded / 255.0
            padded = np.expand_dims(padded, axis=0)
            padded = np.expand_dims(padded, axis=-1)
            pred = model.predict(padded)
            pred = np.argmax(pred, axis=1)
            results.append(real_labels[np.where(label_class == pred[0])[0][0]])
