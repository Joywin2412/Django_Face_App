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
import json
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
textbot = load('myapp/textbot.joblib')
tokenizer_obj_text = load('myapp/tokenizertext.joblib')

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
def Chat(request):
    return render(request,'chatbot.html')
def Static(request):
    return render(request,'index.html')
#to capture video class
def ChatAction(request):
    data = json.loads(request.body.decode('utf-8'))
    input_data = data.get('data', '')
    print(input_data)
    text = input_data
    dict = {'recommend': 0,
 'advantages_of_certain': 1,
 'goodbye': 2,
 'name': 3,
 'greeting': 4,
 'delete': 5,
 'order_status': 6,
 'payment_methods': 7,
 'view_categories': 8,
 'product_details': 9,
 'discounts': 10,
 'customer_support': 11,
 'wishlist': 12,
 'order_tracking': 13,
 'returns': 14,
 'payment_issues': 15,
 'shipping_info': 16,
 'product_availability': 17,
 'recommend_similar': 18,
 'account_creation': 19,
 'order_confirmation': 20,
 'stock_notification': 21,
 'comparison': 22,
 'view_products': 23}
    dictionary2 = {'recommend': ['Welcome Sir, I recommend you this product. People have said these things about this product: I hope you like it.'],
 'advantages_of_certain': ["These are the features you'll get after buying this product."],
 'goodbye': ['Bye', 'Take care'],
 'name': ['My name is Accent. Developers named me that.',
  'I am Accent. Ready to be at your service.'],
 'greeting': ['Hi there. I am Accent! Ready to help.', 'Hello', 'Hi :)'],
 'delete': ['Extremely sorry to hear that! We will improve in the future. If you are unhappy with our service, deleting the account is possible via your profile accessed by clicking on the top of your page.'],
 'order_status': ['Sure, I can help you with that. Please provide me with your order number.'],
 'payment_methods': ['We accept various payment methods including credit cards, PayPal, and more.'],
 'view_categories': ['We have a wide range of categories including electronics, clothing, accessories, and more. Which category are you interested in?'],
 'product_details': ['Certainly! PRODUCT_NAME is a top-rated product known for its quality and performance. It features XYZ specifications and comes with ABC features. Customers have been really satisfied with its performance.'],
 'discounts': ["Yes, we currently have some exciting discounts and offers on various products. Make sure to check out our 'Deals' section for more details."],
 'customer_support': ["Of course, I'm here to assist you! Please provide me with more details about your issue, and I'll do my best to help. If it's a complex matter, I can also connect you to our customer support team."],
 'wishlist': ["Our wishlist feature allows you to save products you're interested in for future reference. You can add items to your wishlist by clicking the 'Add to Wishlist' button on the product page."],
 'order_tracking': ["Sure! Please provide me with your order number, and I'll check the status of your order for you."],
 'returns': ["I'm sorry to hear that. Our return policy allows you to return products within 30 days of purchase. Please visit our 'Returns' page for more information or contact our customer support for assistance with a damaged item."],
 'payment_issues': ["I apologize for the inconvenience. Payment issues can sometimes occur due to various reasons. Please double-check your payment information and try again. If the issue persists, don't hesitate to contact our customer support for further assistance."],
 'shipping_info': ["We offer various shipping options including standard and express. Shipping times and charges depend on your location and the chosen shipping method. You can find more details on our 'Shipping' page."],
 'product_availability': ['Let me check the current stock for you. One moment... Yes, PRODUCT_NAME is currently in stock and available for purchase.'],
 'recommend_similar': ['Certainly! If you like PRODUCT_NAME, you might also be interested in these similar products: PRODUCT_1, PRODUCT_2, and PRODUCT_3.'],
 'account_creation': ["Creating an account is easy! Just click on the 'Sign Up' button at the top of the page and follow the instructions to create your account."],
 'order_confirmation': ["Order confirmations are usually sent to the email address associated with your account. If you haven't received it, please check your spam folder. If you still can't find it, feel free to contact us, and we'll assist you."],
 'stock_notification': ["Certainly! We can notify you via email when PRODUCT_NAME is back in stock. Just provide us with your email address, and we'll keep you updated."],
 'comparison': ['This product is better than another'],
 'view_products': ['These are the list of products available under this category']}
    
    test_lines = clean_text([text])
    print(test_lines)
    
    text = []
    # text.append(str(TextBlob(test_lines[0]).correct()))
    text.append(test_lines[0])
    print(text)
    test_sequences = tokenizer_obj_text.texts_to_sequences(text)
    print(test_sequences)
    # consider = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
    test_review_pad = pad_sequences(test_sequences, maxlen=15, padding='post')
    print(test_review_pad)

    pred = textbot.predict([test_review_pad])
    pred*=100
    pred[0] = np.array(pred[0])
    print(pred)
    i = np.argmax(pred[0])
    print(i)
    inverse_dict = {value: key for key, value in dict.items()}
    print(inverse_dict[i])
    ourResult = random.choice(dictionary2[inverse_dict[i]])
    return JsonResponse({'chat_response':ourResult},status = 200)

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