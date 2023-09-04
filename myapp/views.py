import re
import nltk
import time
from django.http import JsonResponse

import string
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from django.http import HttpResponse
from django.http import FileResponse
import os
from django.shortcuts import render
from .models import *
from django.core.mail import EmailMessage
from django.views.decorators import gzip
from django.http import StreamingHttpResponse
import cv2
import threading
from PIL import Image
import speech_recognition as sr
import pyttsx3
from joblib import load
from keras_preprocessing.sequence import pad_sequences
import random
import numpy as np
dictionary2 = {'search_product': ['Sure, I can help you find PRODUCT_NAME. Please wait a moment while I search for it.'],
 'product_found': ['Great! PRODUCT_NAME has been added to your cart. What would you like to do next?'],
 'add_to_cart': ['Adding PRODUCT_NAME to your cart.'],
 'add_to_wishlist': ['PRODUCT_NAME has been added to your wishlist. You can view it later in your wishlist.'],
 'view_cart': ['Here are the items in your cart: [List of items].'],
 'checkout': ["Sure, let's proceed to checkout. Please confirm your shipping address and payment method."],
 'make_payment': ['We accept various payment methods, including credit cards, PayPal, and more. You can choose your preferred payment option during checkout.'],
 'order_confirmation': ["Order confirmations are usually sent to the email address associated with your account. If you haven't received it, please check your spam folder. If you still can't find it, feel free to contact us, and we'll assist you."],
 'goodbye': ['Goodbye! If you have any more questions in the future, feel free to ask.']}

dict = {'search_product': 0,
 'product_found': 1,
 'add_to_cart': 2,
 'add_to_wishlist': 3,
 'view_cart': 4,
 'checkout': 5,
 'make_payment': 6,
 'order_confirmation': 7,
 'goodbye': 8}

bot = load('myapp/bot.joblib')
tokenizer_obj = load('myapp/tokenizer.joblib')
recognizer = sr.Recognizer()


@gzip.gzip_page
def Home(request):
    try:
        cam = VideoCamera()
        return StreamingHttpResponse(gen(cam), content_type="multipart/x-mixed-replace;boundary=frame")
    except:
        pass
    # return 
def audio_generator():
        with open("now.wav", 'rb') as audio_file:
            while True:
                data = audio_file.read(1024)
                if not data:
                    break
                yield data

def Audio(request):
    file = 'now.wav'  
    location = "myapp/now.wav"
    try:
        os.remove(location)
    except:
        print('already')
    with sr.Microphone() as source:
    # print("Listening..")
        audio = recognizer.listen(source)
        print(audio)
        command = recognizer.recognize_google(audio)
        print("You said: ", command)
        engine = pyttsx3.init()
        response = func(command)
        print(response)
        # engine.say(response)
        timestamp = int(time.time())
        audio_file_name = f'myapp/now.wav'
        engine.save_to_file(response,audio_file_name)
        engine.runAndWait()

        # Return the timestamp as JSON response
        return JsonResponse({'timestamp': timestamp})
        # response = -1

def Static(request):
    return render(request,'index.html')
#to capture video class
class VideoCamera(object):
    def __init__(self):
        self.video = cv2.VideoCapture(0)
        (self.grabbed, self.frame) = self.video.read()
        frame_width = int(self.video.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(self.video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.overlay_image = cv2.imread("myapp/Angry.png", cv2.IMREAD_UNCHANGED)
        self.overlay_image_resized = cv2.resize(self.overlay_image, (int(frame_width), int(frame_height)))
        self.alpha_channel = self.overlay_image_resized[:, :, 3] / 255.0
        self.frame = cv2.resize(self.frame,(640,480))
        threading.Thread(target=self.update, args=()).start()

    def __del__(self):
        self.video.release()

    def get_frame(self):
        image = self.frame
        # image.save('Now.jpg')
        
        image_now = Image.fromarray(cv2.cvtColor(self.frame, cv2.COLOR_BGR2RGB))

        # Save the PIL Image
        image_now.save('Now.jpg')
        

        _, jpeg = cv2.imencode('.jpg', image)
        return jpeg.tobytes()

    def update(self):
        while True:
            

                
            
            (self.grabbed, self.frame) = self.video.read()
            # for c in range(0, 3):
            # self.frame[:, :] = self.frame[:, :] * (1 - self.alpha_channel) + self.overlay_image_resized[:, :] * self.alpha_channel

def gen(camera):
    while True:
        frame = camera.get_frame()
        # frame_width = int(camera.get(cv2.CAP_PROP_FRAME_WIDTH))
        # frame_height = int(camera.get(cv2.CAP_PROP_FRAME_HEIGHT))
        # overlay_image_resized = cv2.resize(overlay_image, (int(frame_width), int(frame_height)))

        
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')
        

def clean_text(text):
    all_reviews = []
    pattern = re.compile('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')
    for i in range(0,len(text)):
        now_text = text[i]
        try:
            now_text = now_text.lower()
            now_text = pattern.sub('', now_text)
            now_text = re.sub(r"[,.\"!@#$%^&*(){}?/;`~:<>+=-]", "", now_text)
            now_text = re.sub(r"i'm", "i am", now_text)
            now_text = re.sub(r"he's", "he is", now_text)
            now_text = re.sub(r"she's", "she is", now_text)
            now_text = re.sub(r"that's", "that is", now_text)        
            now_text = re.sub(r"what's", "what is", now_text)
            now_text = re.sub(r"where's", "where is", now_text) 
            now_text = re.sub(r"\'ll", " will", now_text)  
            now_text = re.sub(r"\'ve", " have", now_text)  
            now_text = re.sub(r"\'re", " are", now_text)
            now_text = re.sub(r"\'d", " would", now_text)
            now_text = re.sub(r"\'ve", " have", now_text)
            now_text = re.sub(r"won't", "will not", now_text)
            now_text = re.sub(r"don't", "do not", now_text)
            now_text = re.sub(r"did't", "did not", now_text)
            now_text = re.sub(r"can't", "can not", now_text)
            now_text = re.sub(r"it's", "it is", now_text)
            now_text = re.sub(r"couldn't", "could not", now_text)
            now_text = re.sub(r"have't", "have not", now_text)
            tokens = word_tokenize(now_text)
            table = str.maketrans('', '', string.punctuation)
            stripped = [w.translate(table) for w in tokens]
            words = [word for word in stripped if word.isalpha()]
            
            PS = PorterStemmer()
            words = [PS.stem(w) for w in words ]
            words = ' '.join(words)
            all_reviews.append(words)
            print(words)
            cnt = cnt+1
        except:
            
            continue
    return all_reviews

# all_reviews = clean_text(text)

def predict_sarcasm(text):
    # Send an array of 1 value
    test_lines = clean_text(text)
    test_sequences = tokenizer_obj.texts_to_sequences(test_lines)
    # print(test_sequences)
    test_review_pad = pad_sequences(test_sequences, maxlen=15, padding='post')
    
    pred = bot.predict(test_review_pad)
    pred*=100
    print(pred)
    return pred

def func(command):
    # text ="Add to wishlist"
# text = str(TextBlob(text).correct())
    pred  =predict_sarcasm([command])
    pred[0] = np.array(pred[0])
    i = np.argmax(pred[0])
    print(i)
    inverse_dict = {value: key for key, value in dict.items()}

    # print(inverse_dict[i])
    ourResult = random.choice(dictionary2[inverse_dict[i]])
    print(ourResult)
    return ourResult