import cognitive_face as CF
import urllib3

# disable insecure warnings
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

KEY = '6f850447110646e68b24b1365173e344'
CF.Key.set(KEY)

BASE_URL = 'https://westcentralus.api.cognitive.microsoft.com/face/v1.0'  # Replace with your regional Base URL
CF.BaseUrl.set(BASE_URL)

img_url = 'https://raw.githubusercontent.com/Microsoft/Cognitive-Face-Windows/master/Data/detection1.jpg'
faces = CF.face.detect(img_url,face_id=True, landmarks=False, attributes='emotion')
print(faces[0]['faceAttributes'])
detectedFaceId1 = faces[0]['faceId']

# img_url = "images/verify.jpeg"
# faces = CF.face.detect(img_url,face_id=True, landmarks=False, attributes='emotion')
# detectedFaceId2 = faces[0]['faceId']
#
# # verify faces are the same
# result = CF.face.verify(detectedFaceId1,detectedFaceId2)
# print(result)