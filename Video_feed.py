
#Necessary lilbraries
import cv2 as cv
import numpy as np

face_cascade = cv.CascadeClassifier('haarcascades/haarcascade_frontalface_default.xml')
eye_cascade = cv.CascadeClassifier('haarcascades/haarcascade_eye.xml')

#Load the camera for capture
capture = cv.VideoCapture(0)

while True:
	ret, img = capture.read()
	#Flip the feed to get a mirror image	
	img = cv.flip(img, 1)
	gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
	faces= face_cascade.detectMultiScale(gray, 1.3, 5)

	for (x,y,w,h) in faces:
		cv.rectangle(img, (x,y), (x+w, y+h), (255,0,0), 2)
		roi_gray = gray[y:y+h, x:x+w]
		roi_color = img[y:y+h, x:x+w]

		eyes = eye_cascade.detectMultiScale(roi_gray)

		for (ex,ey,ew,eh) in eyes:
			cv.rectangle(roi_color, (ex,ey), (ex+ew, ey+eh), (0, 255,0), 2)

	cv.imshow('Video Feed',img)
	
	if (cv.waitKey(1) & 0xFF == ord('q')):
		break

capture.release()
cv.destroyAllWindows()


