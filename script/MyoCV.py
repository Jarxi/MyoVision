from __future__ import print_function
import numpy as np
from os.path import join, dirname
from watson_developer_cloud import SpeechToTextV1
from watson_developer_cloud.websocket import RecognizeCallback
import cv2
import myo as libmyo; libmyo.init("/Users/chenyihan/Desktop/USC/hackathon/LAHacks2018/FeelTheWorld/MyoSDK/myo.framework")
import time
import sys
import urllib3
import cognitive_face as CF
from PIL import Image
import operator
import subprocess
from os import system

HEIGHT = 720
WIDTH = 1280

speech_to_text = SpeechToTextV1(
    username='1d91dc2b-0389-4993-86b4-89d8a1bf8d57',
    password='zx4IBtfMN8TM',
    url='https://stream.watsonplatform.net/speech-to-text/api')

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
                degree = locateObject(image, startX, endX, startY, endY)
                y = startY - 15 if startY - 15 > 15 else startY + 15
                cv2.putText(image, label, (startX, y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[idx], 2)
                return degree

# return 1 if in center, 0 if not
def locateObject(image, startX, endX, startY, endY):
    print(startX,endX,startY,endY)
    height = abs(startY - endY)
    width = abs(startX - endX)
    targetTopY = ((HEIGHT - height)/2)*0.8
    targetBotY = (HEIGHT - targetTopY)*1.2
    targetLeftX = ((WIDTH - width)/2)*0.8
    targetRightX = (WIDTH - targetLeftX)*1.2
    print(targetLeftX,targetRightX,targetTopY,targetBotY)
    # print out center box
    cv2.rectangle(image,(int(targetLeftX),int(targetTopY)),(int(targetRightX),int(targetBotY)),(255,0,0),2)
    if startX >= targetLeftX and endX <= targetRightX and startY >= targetTopY and endY <= targetBotY:
        print("center")
        return 1
    else:
        print("not center")
        return 0

def detectFace(image, emotion, myo):
    path = "/Library/Frameworks/Python.framework/Versions/3.5/lib/python3.5/site-packages/cv2/data/haarcascade_frontalface_default.xml"
    face_cascade = cv2.CascadeClassifier(path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x, y, w, h) in faces:
        cv2.rectangle(image,(x,y),(x+w,y+h),(255,0,0),2)
        cv2.putText(image, emotion, (x, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)


# vibrate the Myo band based on given interval
def vibrate(myo, interval):
    myo.vibrate("short")
    time.sleep(float(interval))

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
        faceRectangle = face['faceRectangle']
        x = faceRectangle['left']
        y = faceRectangle['top']
        w = faceRectangle['width']
        h = faceRectangle['height']
        emotions = face['faceAttributes']['emotion']
        emotion = max(emotions.items(), key=operator.itemgetter(1))[0]
        print(emotion)
        return emotion

def record():
    record_cmd = 'sox -b 32 -e unsigned-integer -r 96k -c 2 -d --clobber --buffer $((96000*2*10)) input.wav trim 0 5'
    system(record_cmd)

def getAudioText():
    with open(join(dirname(__file__), 'input.wav'),
          'rb') as audio_file:
        result =  speech_to_text.recognize(
                    audio=audio_file,
                    content_type='audio/wav',
                    timestamps=True,
                    word_confidence=True)
        str = ''
        for obj in result['results']:
            str += obj['alternatives'][0]['transcript']
        return str

def getTarget(myo):
    system('say What do you want to find?')
    myo.vibrate('short')
    record()
    rawText = getAudioText()
    print(rawText)
    if rawText.lower().find('bottle') != -1:
        return 'bottle'
    elif rawText.lower().find('emotion') != -1:
        return 'emotion'
    elif rawText.lower().find('spotify') != -1:
        return 'spotify'
    elif rawText.lower().find('itunes') != -1:
        return 'itunes'
    elif rawText.lower().find('power point') != -1:
        return 'powerpoint'
    else:
        return ''

def main():

    feed = libmyo.device_listener.Feed()
    hub = libmyo.Hub()
    hub.run(1000, feed)
    try:
        myo = feed.wait_for_single_device(timeout=10.0)  # seconds
        if not myo:
            print("No Myo connected after 10 seconds.")
            system('say No Myo connected after 10 seconds')
            sys.exit()

        # on connect
        if myo.connected:
            myo.vibrate("short")
            myo.vibrate("medium")

        while 1:
            cap = cv2.VideoCapture(0)
            target = getTarget(myo)
            print(target)

            #target = input("What do you want to find? ")
            # for limiting emotion analysis rate

            if target == 'bottle':
                #while myo.pose != "double_tap": {} # wait for command

                system('say Start looking for bottle')

                # start finding
                myo.vibrate("short")
                myo.vibrate("short")

                while cap.isOpened() and hub.running and myo.connected:
                    ret, frame = cap.read()
                    #if target == 'person':
                    #    detectFace(frame)
                    degree = detectObject(frame, target)
                    if (degree):
                        myo.vibrate("short")
                    cv2.imshow('frame',frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                    if myo.pose == 'double_tap':
                        print("Quit finding")
                        system('say quit finding')
                        break

                cap.release()
                cv2.destroyAllWindows()

            elif target == 'emotion': 
                counter = 0
                emotion = 'neutral'
                system('say start detecting emotion')
                while(cap.isOpened()):
                    ret, frame = cap.read()
                    detectFace(frame, emotion, myo)
                    if counter % 15 == 0:
                        emotion = detectEmotion(frame)
                        counter = 0
                        if emotion == 'surprise' or emotion == 'anger':
                            myo.vibrate('long')
                            myo.vibrate('short')
                            myo.vibrate('long')
                        elif emotion == 'neutral' or emotion == 'contempt':
                            myo.vibrate('short')
                            myo.vibrate('medium')
                            myo.vibrate('short')
                        elif emotion == 'disgust' or emotion == 'fear' or emotion == 'sadness':
                            myo.vibrate('short')
                            myo.vibrate('medium')
                            myo.vibrate('medium')
                    cv2.imshow('frame',frame)
                    counter = counter + 1
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                    if myo.pose == 'double_tap':
                        print("Quit finding")
                        system('say quit finding')
                        break

                cap.release()
                cv2.destroyAllWindows()
            elif target == 'spotify':
                system('say start spotify')
                subprocess.call(
                    ["/usr/bin/open", "-n", "-a", "/Applications/Spotify.app"]
                    )
                return 1
            elif target == 'itunes':
                system('say start itunes')
                subprocess.call(
                    ["/usr/bin/open", "-n", "-a", "/Applications/iTunes.app"]
                    )
                return 1
            elif target == 'powerpoint':
                system('say start PowerPoint')
                subprocess.call(
                    ["/usr/bin/open", "-n", "-a", "/Applications/Microsoft PowerPoint.app"]
                    )
                return 1
            else:
                system('say Sorry, I can not understand')
    except KeyboardInterrupt:
        system('say Thanks for using this program')
        print("Quitting...")
    finally:
        hub.shutdown()

    

main()
