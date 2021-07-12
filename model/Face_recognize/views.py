from django.shortcuts import render, redirect
from django.http.response import StreamingHttpResponse
from model.Face_recognize.camera import FaceDetect
import cv2, os
from PIL import Image
from pathlib import Path
from django.conf import settings

def index(request):
    return render(request, 'index.html')

def recognition_home(request):
    return render(request, 'recognition_home.html')

def create_dataset(request):
    userName = request.POST['userName']

    name = userName

    video = cv2.VideoCapture(0)

    cv2.namedWindow("test")

    img_counter = 0

    Path(os.path.sep.join([settings.BASE_DIR,"model/dataset/{}".format(name)])).mkdir(parents=True, exist_ok=True)
    faceCascade = cv2.CascadeClassifier(os.path.sep.join([settings.BASE_DIR, "model\\face_detection_model\\haarcascade_frontalface_default.xml"]))
    while True:
        ret, frame = video.read()
        #copy the original image
        if frame is None:
            break
        originalImg = frame.copy()
        # convert color img to gray scale img
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = faceCascade.detectMultiScale(gray,scaleFactor=1.1,minNeighbors = 5,minSize=(30,30),flags = cv2.CASCADE_SCALE_IMAGE)
        #Draw a rectangle around the faces
        for(x, y, w, h) in faces:
            cv2.rectangle(frame, (x,y),(x+w, y+h), (0,255, 0), 2)

        # To show face recognition window


    
        cv2.imshow("test", frame)

        k = cv2.waitKey(1)
        if k%256 == 27:
            # ESC pressed
            print("Escape hit, closing...")
            break
        elif k%256 == 32:
            # SPACE pressed
            img = os.path.sep.join([settings.BASE_DIR,"model/dataset/"+ name +"/{}.jpg".format(img_counter)])
            cv2.imwrite(img, originalImg)
            print("{} written!".format(img))
            img_counter += 1
            if img_counter == 6:
                break

    video.release()

    cv2.destroyAllWindows()

    return render(request, 'recognition_home.html')



def home(request):
    return render(request, 'index.html')

def recognition(request):
	return render(request, 'recognition.html')



# For Face Recognition 
def gen(camera):
	while True:
		frame = camera.get_frame()
		yield (b'--frame\r\n'
				b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')
        
def facecam_feed(request):
	return StreamingHttpResponse(gen(FaceDetect()),
					content_type='multipart/x-mixed-replace; boundary=frame') 
