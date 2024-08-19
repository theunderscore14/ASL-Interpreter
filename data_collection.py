import numpy as np
import cv2
import os
import string

if not os.path.exists("ASL_Datasets"):
    os.makedirs("ASL_Datasets")
if not os.path.exists("ASL_Datasets/Train"):
    os.makedirs("ASL_Datasets/Train")
if not os.path.exists("ASL_Datasets/Test"):
    os.makedirs("ASL_Datasets/Test")

for i in range(10):
    if not os.path.exists("ASL_Datasets/Train/" + str(i)):
        os.makedirs("ASL_Datasets/Train/"+str(i))
    if not os.path.exists("ASL_Datasets/Test/" + str(i)):
        os.makedirs("ASL_Datasets/Test/"+str(i))

for i in string.ascii_uppercase:
    if not os.path.exists("ASL_Datasets/Train/" + i):
        os.makedirs("ASL_Datasets/Train/" + str(i))
    if not os.path.exists("ASL_Datasets/Test/" + i):
        os.makedirs("ASL_Datasets/Test/" + str(i))

# Train or Test
mode = 'Train'
directory = 'ASL_Datasets/' + mode + '/'

cap = cv2.VideoCapture(0)

while True:
    _, asl_det = cap.read()
    asl_det = cv2.flip(asl_det, flipCode=1)

    count = {
        'zero': len(os.listdir(directory + "/0")),
        'one': len(os.listdir(directory + "/1")),
        'two': len(os.listdir(directory + "/2")),
        'three': len(os.listdir(directory + "/3")),
        'four': len(os.listdir(directory + "/4")),
        'five': len(os.listdir(directory + "/5")),
        'six': len(os.listdir(directory + "/6")),
        'seven': len(os.listdir(directory + "/7")),
        'eight': len(os.listdir(directory + "/8")),
        'nine': len(os.listdir(directory + "/9")),
        'a': len(os.listdir(directory + "/A")),
        'b': len(os.listdir(directory + "/B")),
        'c': len(os.listdir(directory + "/C")),
        'd': len(os.listdir(directory + "/D")),
        'e': len(os.listdir(directory + "/E")),
        'f': len(os.listdir(directory + "/F")),
        'g': len(os.listdir(directory + "/G")),
        'h': len(os.listdir(directory + "/H")),
        'i': len(os.listdir(directory + "/I")),
        'j': len(os.listdir(directory + "/J")),
        'k': len(os.listdir(directory + "/K")),
        'l': len(os.listdir(directory + "/L")),
        'm': len(os.listdir(directory + "/M")),
        'n': len(os.listdir(directory + "/N")),
        'o': len(os.listdir(directory + "/O")),
        'p': len(os.listdir(directory + "/P")),
        'q': len(os.listdir(directory + "/Q")),
        'r': len(os.listdir(directory + "/R")),
        's': len(os.listdir(directory + "/S")),
        't': len(os.listdir(directory + "/T")),
        'u': len(os.listdir(directory + "/U")),
        'v': len(os.listdir(directory + "/V")),
        'w': len(os.listdir(directory + "/W")),
        'x': len(os.listdir(directory + "/X")),
        'y': len(os.listdir(directory + "/Y")),
        'z': len(os.listdir(directory + "/Z"))
    }

    cv2.putText(asl_det, "ZERO : " + str(count['zero']), (10, 70), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 0), 1)
    cv2.putText(asl_det, "ONE : " + str(count['one']), (10, 80), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 0), 1)
    cv2.putText(asl_det, "TWO : " + str(count['two']), (10, 90), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 0), 1)
    cv2.putText(asl_det, "THREE : " + str(count['three']), (10, 100), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 0), 1)
    cv2.putText(asl_det, "FOUR : " + str(count['four']), (10, 110), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 0), 1)
    cv2.putText(asl_det, "FIVE : " + str(count['five']), (10, 120), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 0), 1)
    cv2.putText(asl_det, "SIX : " + str(count['six']), (10, 130), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 0), 1)
    cv2.putText(asl_det, "SEVEN : " + str(count['seven']), (10, 140), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 0), 1)
    cv2.putText(asl_det, "EIGHT : " + str(count['eight']), (10, 150), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 0), 1)
    cv2.putText(asl_det, "NINE : " + str(count['nine']), (10, 160), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 0), 1)
    cv2.putText(asl_det, "A : " + str(count['a']), (10, 170), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 0), 1)
    cv2.putText(asl_det, "B : " + str(count['b']), (10, 180), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 0), 1)
    cv2.putText(asl_det, "C : " + str(count['c']), (10, 190), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 0), 1)
    cv2.putText(asl_det, "D : " + str(count['d']), (10, 200), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 0), 1)
    cv2.putText(asl_det, "E : " + str(count['e']), (10, 210), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 0), 1)
    cv2.putText(asl_det, "F : " + str(count['f']), (10, 220), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 0), 1)
    cv2.putText(asl_det, "G : " + str(count['g']), (10, 230), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 0), 1)
    cv2.putText(asl_det, "H : " + str(count['h']), (10, 240), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 0), 1)
    cv2.putText(asl_det, "I : " + str(count['i']), (10, 250), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 0), 1)
    cv2.putText(asl_det, "J : " + str(count['j']), (10, 260), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 0), 1)
    cv2.putText(asl_det, "K : " + str(count['k']), (10, 270), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 0), 1)
    cv2.putText(asl_det, "L : " + str(count['l']), (10, 280), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 0), 1)
    cv2.putText(asl_det, "M : " + str(count['m']), (10, 290), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 0), 1)
    cv2.putText(asl_det, "N : " + str(count['n']), (10, 300), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 0), 1)
    cv2.putText(asl_det, "O : " + str(count['o']), (10, 310), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 0), 1)
    cv2.putText(asl_det, "P : " + str(count['p']), (10, 320), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 0), 1)
    cv2.putText(asl_det, "Q : " + str(count['q']), (10, 330), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 0), 1)
    cv2.putText(asl_det, "R : " + str(count['r']), (10, 340), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 0), 1)
    cv2.putText(asl_det, "S : " + str(count['s']), (10, 350), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 0), 1)
    cv2.putText(asl_det, "T : " + str(count['t']), (10, 360), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 0), 1)
    cv2.putText(asl_det, "U : " + str(count['u']), (10, 370), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 0), 1)
    cv2.putText(asl_det, "V : " + str(count['v']), (10, 380), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 0), 1)
    cv2.putText(asl_det, "W : " + str(count['w']), (10, 390), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 0), 1)
    cv2.putText(asl_det, "X : " + str(count['x']), (10, 400), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 0), 1)
    cv2.putText(asl_det, "Y : " + str(count['y']), (10, 410), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 0), 1)
    cv2.putText(asl_det, "Z : " + str(count['z']), (10, 420), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 0), 1)

    x1 = int(0.5*asl_det.shape[1])
    y1 = 10
    x2 = asl_det.shape[1]-10
    y2 = int(0.5*asl_det.shape[1])

    cv2.rectangle(asl_det, (220-1, 9), (620 + 1, 419), (255, 0, 0), 1)

    col = asl_det[10:410, 220:520]
    cv2.imshow("Frame", asl_det)

    col = cv2.cvtColor(col, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(col, (5, 5), 2)

    th3 = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    ret, roi = cv2.threshold(th3, 70, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    cv2.imshow("TRAIN", roi)

    interrupt = cv2.waitKey(10)
    if interrupt & 0xFF == 27:  # Escape Key (Esc)
        break
    if interrupt & 0xFF == ord('0'):
        cv2.imwrite(directory + '0/' + str(count['zero']) + '.jpg', roi)
    if interrupt & 0xFF == ord('1'):
        cv2.imwrite(directory + '1/' + str(count['one']) + '.jpg', roi)
    if interrupt & 0xFF == ord('2'):
        cv2.imwrite(directory + '2/' + str(count['two']) + '.jpg', roi)
    if interrupt & 0xFF == ord('3'):
        cv2.imwrite(directory + '3/' + str(count['three']) + '.jpg', roi)
    if interrupt & 0xFF == ord('4'):
        cv2.imwrite(directory + '4/' + str(count['four']) + '.jpg', roi)
    if interrupt & 0xFF == ord('5'):
        cv2.imwrite(directory + '5/' + str(count['five']) + '.jpg', roi)
    if interrupt & 0xFF == ord('6'):
        cv2.imwrite(directory + '6/' + str(count['six']) + '.jpg', roi)
    if interrupt & 0xFF == ord('7'):
        cv2.imwrite(directory + '7/' + str(count['seven']) + '.jpg', roi)
    if interrupt & 0xFF == ord('8'):
        cv2.imwrite(directory + '8/' + str(count['eight']) + '.jpg', roi)
    if interrupt & 0xFF == ord('9'):
        cv2.imwrite(directory + '9/' + str(count['nine']) + '.jpg', roi)
    if interrupt & 0xFF == ord('a'):
        cv2.imwrite(directory + 'A/' + str(count['a']) + '.jpg', roi)
    if interrupt & 0xFF == ord('b'):
        cv2.imwrite(directory + 'B/' + str(count['b']) + '.jpg', roi)
    if interrupt & 0xFF == ord('c'):
        cv2.imwrite(directory + 'C/' + str(count['c']) + '.jpg', roi)
    if interrupt & 0xFF == ord('d'):
        cv2.imwrite(directory + 'D/' + str(count['d']) + '.jpg', roi)
    if interrupt & 0xFF == ord('e'):
        cv2.imwrite(directory + 'E/' + str(count['e']) + '.jpg', roi)
    if interrupt & 0xFF == ord('f'):
        cv2.imwrite(directory + 'F/' + str(count['f']) + '.jpg', roi)
    if interrupt & 0xFF == ord('g'):
        cv2.imwrite(directory + 'G/' + str(count['g']) + '.jpg', roi)
    if interrupt & 0xFF == ord('h'):
        cv2.imwrite(directory + 'H/' + str(count['h']) + '.jpg', roi)
    if interrupt & 0xFF == ord('i'):
        cv2.imwrite(directory + 'I/' + str(count['i']) + '.jpg', roi)
    if interrupt & 0xFF == ord('j'):
        cv2.imwrite(directory + 'J/' + str(count['j']) + '.jpg', roi)
    if interrupt & 0xFF == ord('k'):
        cv2.imwrite(directory + 'K/' + str(count['k']) + '.jpg', roi)
    if interrupt & 0xFF == ord('l'):
        cv2.imwrite(directory + 'L/' + str(count['l']) + '.jpg', roi)
    if interrupt & 0xFF == ord('m'):
        cv2.imwrite(directory + 'M/' + str(count['m']) + '.jpg', roi)
    if interrupt & 0xFF == ord('n'):
        cv2.imwrite(directory + 'N/' + str(count['n']) + '.jpg', roi)
    if interrupt & 0xFF == ord('o'):
        cv2.imwrite(directory + 'O/' + str(count['o']) + '.jpg', roi)
    if interrupt & 0xFF == ord('p'):
        cv2.imwrite(directory + 'P/' + str(count['p']) + '.jpg', roi)
    if interrupt & 0xFF == ord('q'):
        cv2.imwrite(directory + 'Q/' + str(count['q']) + '.jpg', roi)
    if interrupt & 0xFF == ord('r'):
        cv2.imwrite(directory + 'R/' + str(count['r']) + '.jpg', roi)
    if interrupt & 0xFF == ord('s'):
        cv2.imwrite(directory + 'S/' + str(count['s']) + '.jpg', roi)
    if interrupt & 0xFF == ord('t'):
        cv2.imwrite(directory + 'T/' + str(count['t']) + '.jpg', roi)
    if interrupt & 0xFF == ord('u'):
        cv2.imwrite(directory + 'U/' + str(count['u']) + '.jpg', roi)
    if interrupt & 0xFF == ord('v'):
        cv2.imwrite(directory + 'V/' + str(count['v']) + '.jpg', roi)
    if interrupt & 0xFF == ord('w'):
        cv2.imwrite(directory + 'W/' + str(count['w']) + '.jpg', roi)
    if interrupt & 0xFF == ord('x'):
        cv2.imwrite(directory + 'X/' + str(count['x']) + '.jpg', roi)
    if interrupt & 0xFF == ord('y'):
        cv2.imwrite(directory + 'Y/' + str(count['y']) + '.jpg', roi)
    if interrupt & 0xFF == ord('z'):
        cv2.imwrite(directory + 'Z/' + str(count['z']) + '.jpg', roi)

    
cap.release()
cv2.destroyAllWindows()
