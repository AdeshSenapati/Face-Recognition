import cv2
import numpy as np
import face_recognition
import os
import time

path = 'images'
images = []
names = []
myList = os.listdir(path)
print(myList)
for cl in myList:
    curImg = cv2.imread(f'{path}/{cl}')
    images.append(curImg)
    names.append(os.path.splitext(cl)[0])
print(names)

# for images in database
def findEncodings(images):   # to find the encodings of all the images in database
    encodeList = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodeList.append(encode)
    return encodeList


encodeListKnown = findEncodings(images)
print("encoding done......")

# for img in video
cap = cv2.VideoCapture('Bill Gates_ The next outbreak_ We’re not ready _ TED.mp4')

while True:
    success, img = cap.read()
    pTime = 0
    imgSmall = cv2.resize(img, (0, 0), None, 0.25, 0.25)  # resizing to speedup recognition
    imgSmall = cv2.cvtColor(imgSmall, cv2.COLOR_BGR2RGB)

    facesCurFrame = face_recognition.face_locations(imgSmall)
    encodeCurFrame = face_recognition.face_encodings(imgSmall, facesCurFrame)

    for encodeFace, faceLoc in zip(encodeCurFrame, facesCurFrame):
        matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
        faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)  # lowest distance means best match
        # basically facedis will calc the distance between the encodings and the one with low ditance will be the best
        # match and it will return that name from directory of images...
        matchIndex = np.argmin(faceDis)  # taking index of lowest

        if matches[matchIndex]:
            name = names[matchIndex]  # getting the matched name at that index
            y1, x2, y2, x1 = faceLoc
            y1, x2, y2, x1 = y1*4, x2*4, y2*4, x1*4
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(img, name, (300, 70), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)

    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime
    cv2.putText(img, f'FPS:{int(fps)}', (20, 70), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)
    cv2.imshow("Image", img)
    cv2.waitKey(1)


