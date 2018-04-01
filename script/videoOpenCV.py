import numpy as np
import cv2
import urllib3
import cognitive_face as CF
from PIL import Image
import operator

HEIGHT = 720
WIDTH = 1280
cap = cv2.VideoCapture(0)
# cap = cv2.VideoCapture(1)
def detectEmotion(image):
    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

    KEY = '6f850447110646e68b24b1365173e344'
    CF.Key.set(KEY)

    BASE_URL = 'https://westcentralus.api.cognitive.microsoft.com/face/v1.0'  # Replace with your regional Base URL
    CF.BaseUrl.set(BASE_URL)
    # image = image.tostring()
    img = Image.fromarray(image, 'RGB')
    # img_url = 'https://raw.githubusercontent.com/Microsoft/Cognitive-Face-Windows/master/Data/detection1.jpg'
    faces = CF.face.detect(img, face_id=True, landmarks=False, attributes='emotion')
    for face in faces:
        emotions = face['faceAttributes']['emotion']
        emotion = max(emotions.items(), key=operator.itemgetter(1))[0]
        print(emotion)
        return emotion


def detectObject(image, target = 'person'):
    # initialize the list of class labels MobileNet SSD was trained to
    # detect, then generate a set of bounding box colors for each class
    CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
               "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
               "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
               "sofa", "train", "tvmonitor"]

    COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))

    # load our serialized model from disk
    net = cv2.dnn.readNetFromCaffe("MobileNetSSD_deploy.prototxt.txt", "MobileNetSSD_deploy.caffemodel")

    (h, w) = image.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 0.007843,
                                 (300, 300), 127.5)

    '''
    pass the blob through the network and obtain the detections andpredictions
    '''
    net.setInput(blob)
    detections = net.forward()

    # loop over the detections
    for i in np.arange(0, detections.shape[2]):
        # extract the confidence (i.e., probability) associated with the
        # prediction
        confidence = detections[0, 0, i, 2]

        # filter out weak detections by ensuring the `confidence` is
        # greater than the minimum confidence
        if confidence > 0.2 :
            # extract the index of the class label from the `detections`,
            # then compute the (x, y)-coordinates of the bounding box for
            # the object
            idx = int(detections[0, 0, i, 1])
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            if (CLASSES[idx] == target and CLASSES[idx] != 'person'):
                # display the prediction
                label = "{}: {:.2f}%".format(CLASSES[idx], confidence * 100)
                cv2.rectangle(image, (startX, startY), (endX, endY),
                              COLORS[idx], 2)
                locateObject(image, startX, endX, startY, endY)
                y = startY - 15 if startY - 15 > 15 else startY + 15
                cv2.putText(image, label, (startX, y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[idx], 2)

def locateObject(image, startX, endX, startY, endY):
    height = abs(startY - endY)
    width = abs(startX - endX)
    targetTopY = ((HEIGHT - height)/2)*0.8
    targetBotY = (HEIGHT - targetTopY)*1.2
    targetLeftX = ((WIDTH - width)/2)*0.8
    targetRightX = (WIDTH - targetLeftX)*1.2
    cv2.rectangle(image,(int(targetLeftX),int(targetTopY)),(int(targetRightX),int(targetBotY)),(255,144,0),2)
    if startX >= targetLeftX and endX <= targetRightX and startY >= targetTopY and endY <= targetBotY:
        print("center")
    else:
        print("not center")

def detectFace(image, emotion):
    path = "/Library/Frameworks/Python.framework/Versions/3.5/lib/python3.5/site-packages/cv2/data/haarcascade_frontalface_default.xml"
    face_cascade = cv2.CascadeClassifier(path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x, y, w, h) in faces:
        cv2.rectangle(image,(x,y),(x+w,y+h),(255,0,0),2)
        cv2.putText(image, emotion, (x, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        locateObject(image, x, x + w, y, y + h)


def main():
    target = input("What do you want to find? ")
    counter = 0
    emotion = 'neutral'
    while(cap.isOpened()):
        ret, frame = cap.read()
        # detect people's face
        if target == 'emotion':
            detectFace(frame, emotion)
            if counter%20 == 0:
                emotion = detectEmotion(frame)
                counter = 0
        # detect objects
        else:
            detectObject(frame, target)
        cv2.imshow('frame',frame)
        counter = counter + 1
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

main()

