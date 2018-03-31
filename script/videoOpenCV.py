import numpy as np
import cv2


HEIGHT = 720
WIDTH = 1280
cap = cv2.VideoCapture(0)

def detectObject(image, target = 'person'):
    # initialize the list of class labels MobileNet SSD was trained to
    # detect, then generate a set of bounding box colors for each class
    CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
               "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
               "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
               "sofa", "train", "tvmonitor"]

    COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))

    # load our serialized model from disk
    # print("[INFO] loading model...")
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
                # area = (endX - startX) * (endY - startY)
                locateObject(startX, endX, startY, endY)
                y = startY - 15 if startY - 15 > 15 else startY + 15
                cv2.putText(image, label, (startX, y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[idx], 2)

def locateObject(startX, endX, startY, endY):
    print(startX,endX,startY,endY)
    height = abs(startY - endY)
    width = abs(startX - endX)
    targetTopY = ((HEIGHT - height)/2)*0.8
    targetBotY = (HEIGHT - targetTopY)*1.2
    targetLeftX = ((WIDTH - width)/2)*0.8
    targetRightX = (WIDTH - targetLeftX)*1.2
    print(targetLeftX,targetRightX,targetTopY,targetBotY)
    if startX >= targetLeftX and endX <= targetRightX and startY >= targetTopY and endY <= targetBotY:
        print("center")
    else:
        print("not center")

def detectFace(image):
    path = "/Library/Frameworks/Python.framework/Versions/3.5/lib/python3.5/site-packages/cv2/data/haarcascade_frontalface_default.xml"
    face_cascade = cv2.CascadeClassifier(path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x, y, w, h) in faces:
        cv2.rectangle(image,(x,y),(x+w,y+h),(255,0,0),2)
        locateObject(x, x + w, y, y + h)


def main():
    target = input("What do you want to find? ")
    while(cap.isOpened()):
        ret, frame = cap.read()
        if target == 'person':
            detectFace(frame)
        detectObject(frame, target)
        cv2.imshow('frame',frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

main()

