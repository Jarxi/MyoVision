from __future__ import print_function
import json
from os.path import join, dirname
from watson_developer_cloud import SpeechToTextV1
from watson_developer_cloud.websocket import RecognizeCallback
from os import system

speech_to_text = SpeechToTextV1(
    username='1d91dc2b-0389-4993-86b4-89d8a1bf8d57',
    password='zx4IBtfMN8TM',
    url='https://stream.watsonplatform.net/speech-to-text/api')

#print(json.dumps(speech_to_text.list_models(), indent=2))

#print(json.dumps(speech_to_text.get_model('en-US_BroadbandModel'), indent=2))

record_cmd = 'sox -b 32 -e unsigned-integer -r 96k -c 2 -d --clobber --buffer $((96000*2*10)) input.wav trim 0 5'
system(record_cmd)

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
    print (str)
