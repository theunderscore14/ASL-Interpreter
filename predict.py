import numpy as np
import operator
import cv2
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import tensorflow as tf
from tensorflow import keras
from keras.models import model_from_json
import sys, os

json_file = open("model-bw.json", "r")
model_json = json_file.read()
json_file.close()
asl_model = model_from_json(model_json)

asl_model.load_weights("model_bw.weights.h5")
print("Predicting...")

cap = cv2.VideoCapture(0)



while True:
    _, asl_det = cap.read()
    asl_det = cv2.flip(asl_det, 1)

    x1 = int(0.5*asl_det.shape[1])
    y1 = 10
    x2 = asl_det.shape[1]-10
    y2 = int(0.5*asl_det.shape[1])

    cv2.rectangle(asl_det, (220 - 1, 9), (620 + 1, 419), (255, 0, 0), 1)

    col = asl_det[10:410, 220:520]

    col = cv2.cvtColor(col, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(col, (5, 5), 2)

    th3 = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
    ret, roi = cv2.threshold(th3, 70, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    roi = cv2.resize(roi, (64, 64))
    cv2.imshow("test", roi)

    result = asl_model.predict(roi.reshape(1, 64, 64, 1))

    prediction = {
        'ZERO': result[0][0],
        'ONE': result[0][1],
        'TWO': result[0][2],
        'THREE': result[0][3],
        'FOUR': result[0][4],
        'FIVE': result[0][5],
        'SIX': result[0][6],
        'SEVEN': result[0][7],
        'EIGHT': result[0][8],
        'NINE': result[0][9],
        'A': result[0][10],
        'B': result[0][11],
        'C': result[0][12],
        'D': result[0][13],
        'E': result[0][14],
        'F': result[0][15],
        'G': result[0][16],
        'H': result[0][17],
        'I': result[0][18],
        'J': result[0][19],
        'K': result[0][20],
        'L': result[0][21],
        'M': result[0][22],
        'N': result[0][23],
        'O': result[0][24],
        'P': result[0][25],
        'Q': result[0][26],
        'R': result[0][27],
        'S': result[0][28],
        'T': result[0][29],
        'U': result[0][30],
        'V': result[0][31],
        'W': result[0][32],
        'X': result[0][33],
        'Y': result[0][34],
        'Z': result[0][35],
    }

    prediction = sorted(prediction.items(), key=operator.itemgetter(1), reverse=True)

    cv2.putText(asl_det, prediction[0][0], (10, 420), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 255), 1)
    cv2.imshow("Frame", asl_det)

    interrupt = cv2.waitKey(10)
    if interrupt & 0xFF == 27:
        break


cap.release()
cv2.destroyAllWindows()
