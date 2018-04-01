## List of video demo
https://www.youtube.com/playlist?list=PLfDoTodr3Vpb0vD_DKzmMAosTkPPI2XCn
## Inspiration
  Our society is developed in many ways, but support for the visually impaired is far from enough. A blindman can hardly find an object in an unfamiliar environment, nor can he/she read people's facial expression in a daily conversation. Missing the vision is a lot, but luckily, computer can be the eye of the visually impaired. With gesture control, a blindman can easily give command and interact with the world!

## What it does
Feel The World is built with a WebCam and Myo armband. It utilizes computer vision, voice recognition, gesture  recognition to accomplish the following functionalities:

**1)  Find an object**
   User can give a voice command like "I want to find a bottle," and the camera will start detecting if there is a bottle. Meanwhile, the user can move his arm towards different direction, and when a bottle is detected at the center of its vision, it provides a vibrational feedback through Myo to user. Therefore the user will know the direction of object, and simply stretch his hand he will grab it.

**2) Read a handwritten text**
   User can give a voice command to read a written text, which can be either on a screen or paper. The camera will capture the picture and process it, then read the text.

**3) Understand emotion**
  What the visually impaired miss a lot in daily conversation is the knowledge of people's facial expressions and the emotion they imply - this can actually help them communicate better. User can give a voice command to trigger emotion detection, and use the camera to detect a person's face. The person's emotion will be analyzed, and the intensity of emotion (surprise, angry being the strongest, neutral being the least, etc.) will be provided as vibrational feedback with different pattern to user.

**4) Control App**
  User can give a voice command to open music apps like Spotify, iTunes. They can use different gestures to control the volume, skip, pause, resume a track.

## How we built it
  Our central server processes the video data captured by a WebCam, and provide vibrational feedback to Myo. Myo will also read in different gestures.
Specific implementations for each functionality:

**1)** OpenCV with pre-trained model for object recognition. We calculated the size of the object, its distance to center of image, and return a boolean value to indicate if object is centered.

**2)** OpenCV+Microsoft Cognitive Science API+text to speech service built in with Mac.

**3)** OpenCV+Microsoft Face. Different emotions with scores ranging from 0~1.0 will be given, and we choose the highest one, and classify it based on its intensity.

**4)** Myo gesture recognition+built-in script for spotify&iTunes&PowerPoint control
And most importantly, StackOverflow;) 

## Challenges we ran into
  Wifi. 

## Accomplishments that we're proud of
  The object recognition is actually very accurate. For real it CAN help you find the object. Controlling apps with Myo is also pretty cool. 

## What we learned
  Teamwork, respect, brainstorming and thinking in others' shoes.

## What's next for Feel The World
  This project can easily be integrated to Google Home or Amazon Echo because of its voice recognition functionality. It would be cool to let user connect their LED, fridge with a central server through just a armband. This tool is probably what Iron Man is looking for.