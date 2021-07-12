from django.shortcuts import render, redirect
from django.http.response import StreamingHttpResponse
from model.age_and_gender.camera import AgeDetect

def age_gender(request):
	return render(request, 'age_gender.html')


# For Age and gender Prediction
def gen(camera):
	while True:
		frame = camera.get_age()
		yield (b'--frame\r\n'
				b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')
	

def gendercam_feed(request):
	return StreamingHttpResponse(gen(AgeDetect()),
					content_type='multipart/x-mixed-replace; boundary=frame') 
