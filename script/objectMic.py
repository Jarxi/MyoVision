import requests
import matplotlib.pyplot as plt
from PIL import Image

vision_base_url = "https://westcentralus.api.cognitive.microsoft.com/vision/v1.0/"
vision_analyze_url = vision_base_url + "analyze"
subscription_key = "47314dedfdc74d078707c0fcc0e35051"
assert subscription_key

image_path = "emotion_1.jpg"
image_data = open(image_path, "rb").read()

# image_url = "https://upload.wikimedia.org/wikipedia/commons/thumb/1/12/Broadway_and_Times_Square_by_night.jpg/450px-Broadway_and_Times_Square_by_night.jpg"

# set up headers, params, data
headers = {'Ocp-Apim-Subscription-Key': subscription_key,
              "Content-Type": "application/octet-stream"}
params = {'visualFeatures': 'Description'}
response = requests.post(vision_analyze_url,
                           headers=headers,
                           params=params,
                           data=image_data)

response.raise_for_status()
analysis = response.json()
image_caption = analysis["description"]["captions"][0]["text"].capitalize()
image_caption
print(analysis)

image = Image.open(image_path)
plt.imshow(image)
plt.axis("on")
_ = plt.title(image_caption, size="x-large", y=0)
plt.show()
