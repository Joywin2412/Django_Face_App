from django.http import HttpResponse
from django.shortcuts import render
from .models import *
from django.core.mail import EmailMessage
from django.views.decorators import gzip
from django.http import StreamingHttpResponse
import cv2
import threading
from PIL import Image

@gzip.gzip_page
def Home(request):
    try:
        cam = VideoCamera()
        return StreamingHttpResponse(gen(cam), content_type="multipart/x-mixed-replace;boundary=frame")
    except:
        pass
    # return 

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