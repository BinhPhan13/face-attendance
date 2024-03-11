import cv2 as cv
from sys import argv
import detect
import spoof
import recog

enroll = True if len(argv) > 1 and argv[1] == '-e' else False

device = 0
cap = cv.VideoCapture(device)
while cap.isOpened():
    ret, frame = cap.read()
    if not ret: break
    frame = cv.flip(frame, 1)

    key = cv.waitKey(1)
    if ord('q') ==  key: break

    faces = detect.detect_face(frame)
    for (x, y, w, h) in faces:
        cv.rectangle(frame, (x,y), (x+w, y+h), (255,0,0))

    cv.imshow('frame', frame)
    if len(faces) != 1: continue
    ############################################################
    x, y, w, h = faces[0]
    ROI = frame[y:y+h, x:x+w]
    face = ROI.copy()

    eyes = detect.detect_eye(ROI)
    if len(eyes) != 2: continue
    for ex, ey, ew, eh in eyes:
        cv.rectangle(ROI, (ex,ey), (ex+ew, ey+eh), (0,255,0))

    cv.imshow('frame', frame)

    face = detect.align_face(face, eyes)
    cv.imshow('face', face)
    cv.moveWindow('face', 0, 0)

    ############################################################
    real = spoof.isreal(face)
    text = 'Real' if real else 'Fake'
    # print(text)
    cv.putText(
        frame, org=(x,y),
        text=text,
        fontFace=cv.FONT_HERSHEY_SIMPLEX,
        fontScale=1.2, color=(0,0,255), thickness=2
    )
    # if not real: continue

    cv.imshow('frame', frame)
    if enroll and ord('k') == cv.waitKey(1):
        cv.imwrite('test.jpg', face); break

    if not enroll:
        ok = recog.recognize(face)
        text = ok if ok else "Nope!"
        cv.putText(
            frame, org=(x,y+h),
            text=text,
            fontFace=cv.FONT_HERSHEY_SIMPLEX,
            fontScale=1.2, color=(0,255,0), thickness=2
        )
        cv.imshow('frame', frame)

cap.release()
cv.destroyAllWindows()

if enroll: recog.add_data(face)