import os
import io
import json
from azure.cognitiveservices.vision.face import FaceClient
from msrest.authentication import CognitiveServicesCredentials
import requests
from PIL import Image, ImageDraw, ImageFont

"""
Detect if a face shows up in other photos/images
"""

credential = json.load(open('key.json'))
API_KEY = credential["API_KEY"]
ENDPOINT = credential["ENDPOINT"]
face_client = FaceClient(ENDPOINT, CognitiveServicesCredentials(API_KEY))

response_detected_faces = face_client.face.detect_with_stream(
    image=open(r'./tvd.webp', 'rb'),
    detection_model='detection_03',
    recognition_model='recognition_04',  
)
face_ids = [face.face_id for face in response_detected_faces]

img_source = open(r'./damon1.jpeg', 'rb')
response_face_source = face_client.face.detect_with_stream(
    image=img_source,
    detection_model='detection_03',
    recognition_model='recognition_04'    
)
face_id_source = response_face_source[0].face_id

matched_faces = face_client.face.find_similar(
    face_id=face_id_source,
    face_ids=face_ids
)

img = Image.open(open(r'./tvd.webp', 'rb'))
draw = ImageDraw.Draw(img)


for matched_face in matched_faces:
    for face in response_detected_faces:
        if face.face_id == matched_face.face_id:
            rect = face.face_rectangle
            left = rect.left
            top = rect.top
            right = rect.width + left
            bottom = rect.height + top
            draw.rectangle(((left, top), (right, bottom)), outline='green', width=5)
            print(face.face_id)
            
            face_verified = face_client.face.verify_face_to_face(
            face_id1=matched_faces[0].face_id,
            face_id2=face.face_id)
            print(face_verified.is_identical)


img.show()
img.save('img_detected.jpg')