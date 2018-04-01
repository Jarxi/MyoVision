from io import BytesIO
import io
import requests
import time
import matplotlib.pyplot as plt
from PIL import Image
from matplotlib.patches import Polygon

def readText(image):
    vision_base_url = "https://westcentralus.api.cognitive.microsoft.com/vision/v1.0/"
    text_recognition_url = vision_base_url + "RecognizeText"
    subscription_key = "7951e4ef3388446bacf3ba76ecf48e8a"


    image_path = 'output.jpg'
    image_data = open(image_path, "rb").read()

    headers  = {'Ocp-Apim-Subscription-Key': subscription_key,
                'Content-Type': "application/octet-stream"}
    params   = {'handwriting' : True}
    data     = image_data
    response = requests.post(text_recognition_url, headers=headers, params=params, data=data)
    response.raise_for_status()

    operation_url = response.headers["Operation-Location"]

    analysis = {}
    while not "recognitionResult" in analysis:
        response_final = requests.get(response.headers["Operation-Location"], headers=headers)
        analysis = response_final.json()
        time.sleep(1)

    polygons = [(line["boundingBox"], line["text"]) for line in analysis["recognitionResult"]["lines"]]



    plt.figure(figsize=(15,15))

    # image= Image.open(BytesIO(requests.get(image_url).content))
    # image = Image.open(image_path)
    ax = plt.imshow(image)


    sentence = ''
    for polygon in polygons:
        vertices = [(polygon[0][i], polygon[0][i+1]) for i in range(0,len(polygon[0]),2)]
        text= polygon[1]
        sentence = sentence+' '+text
        patch= Polygon(vertices, closed=True,fill=False, linewidth=2, color='y')
        ax.axes.add_patch(patch)
        plt.text(vertices[0][0], vertices[0][1], text, fontsize=20, va="top")
    _ = plt.axis("off")

    print(sentence)
    plt.show()