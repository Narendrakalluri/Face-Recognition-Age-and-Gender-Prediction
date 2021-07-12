from imutils.video import VideoStream
from django.conf import settings
from imutils.video import FPS
import numpy as np
import cv2,os,sys
import imutils
import time
import math

MODEL_MEAN_VALUES = (78.4263377603,87.7689143744,114.895847746)
age_list = ['(0-2)', '(4-6)', '(8-12)', '(15-20)','(21-27)', '(28-32)', '(38-43)', '(48-53)', '(60-100)']
gender_list = ['Male','Female']
conf_threshold = 0.7
padding = 20
faceProto=os.path.sep.join([settings.BASE_DIR, "model\\face_detection_model\\face_deploy.prototxt.txt"])
faceModel=os.path.sep.join([settings.BASE_DIR, "model\\face_detection_model\\face_net.caffemodel"])
ageProto=os.path.sep.join([settings.BASE_DIR, "model\\face_detection_model\\age_deploy.prototxt"])
ageModel=os.path.sep.join([settings.BASE_DIR, "model\\face_detection_model\\age_net.caffemodel"])
genderProto=os.path.sep.join([settings.BASE_DIR, "model\\face_detection_model\\gender_deploy.prototxt"])
genderModel=os.path.sep.join([settings.BASE_DIR, "model\\face_detection_model\\gender_net.caffemodel"])
age_net = cv2.dnn.readNetFromCaffe(ageProto,ageModel)
gender_net = cv2.dnn.readNetFromCaffe(genderProto,genderModel)
net = cv2.dnn.readNetFromCaffe(faceProto,faceModel)

# age and gender detection
class AgeDetect(object):
    
    def __init__(self):
        self.model = cv2.face_LBPHFaceRecognizer.create()
        
        self.model.read(os.path.sep.join([settings.BASE_DIR, 'model\\face_detection_model\\LBPH_trained_data.xml']))
        self.video = VideoStream(src=0).start()
        self.fps = FPS().start()

    def __del__(self):
       	self.video.stream.release()
        cv2.destroyAllWindows()
        

    def get_age(self):
        frame = self.video.read()
        inImg = np.array(frame)
        frame = cv2.flip(inImg,1)
        #frame = inImg
        (h,w) = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(cv2.resize(frame,(300,300)),1.0,
                                     (300,300),(104.0,177.0,123.0))
        net.setInput(blob)
        gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        resized_width,resized_height = (112,92)
        detections = net.forward()
        persons = []
        for i in range(0,detections.shape[2]):
            confidence = detections[0,0,i,2]
            if confidence < 0.5:
                continue
            box = detections[0,0,i,3:7] * np.array([w,h,w,h])
            (startX,startY,endX,endY) = box.astype("int")
            (startX,endX) = np.clip(np.array([startX, endX]), 0,gray.shape[1])
            (startY,endY) = np.clip(np.array([startY, endY]), 0,gray.shape[0])
            y = startY - 10 if startY - 10 > 10 else startY + 10
            cropped = gray[startY:endY,startX:endX]
            resized = cv2.resize(cropped,(resized_width,resized_height))
            face_img = frame[startY:endY, startX:endX]
            fblob = cv2.dnn.blobFromImage(face_img, 1.0, (227, 227), MODEL_MEAN_VALUES, swapRB=False)
            # Predict Gender
            gender_net.setInput(fblob)
            gender_preds = gender_net.forward()
            gender = gender_list[gender_preds[0].argmax()]
            # Predict Age
            age_net.setInput(fblob)
            age_preds = age_net.forward()
            age = age_list[age_preds[0].argmax()]
            # print("Age Range: " + age)

            overlay_text = "%s-%s" % (gender, age)
            confidence = self.model.predict(resized)
            if confidence[1] > 0.5:
                cv2.rectangle(frame, (startX, startY), (endX, endY), (194, 108, 89), 2)
                cv2.putText(frame, 'Info: %s' % (overlay_text),(startX,y),cv2.FONT_HERSHEY_SIMPLEX,0.45,(242, 132, 0),1)
                print(overlay_text)

                #update the FPS counter
        self.fps.update()
        ret, jpeg = cv2.imencode('.jpg', frame)
        return jpeg.tobytes()
