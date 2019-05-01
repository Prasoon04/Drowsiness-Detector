#Necessary libraries
from scipy.spatial import distance as ds
from imutils import face_utils
from playsound import playsound
import cv2 as cv
import numpy as np
import time
import dlib


# If the eye aspect ratio falls below this threshold
#start the counter for the frames
EYE_AR_THRESHOLD = 0.3

#If th eyes are closed for 48 frame, play the alarm.
EYE_AR_CONSECUTIVE_FRAMES = 48

COUNTER = 0

face_cascade = cv.CascadeClassifier('haarcascade_frontalface_default.xml')

def eye_aspect_Ratio(eye):

	A = ds.euclidean(eye[1], eye[5])
	B = ds.euclidean(eye[2], eye[4])
	C = ds.euclidean(eye[0], eye[3])
	ear = (A+B)/(2*C)
	return ear


detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS['left_eye']
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS['right_eye']

feed = cv.VideoCapture(0)
time.sleep(2)

while True:
	ret, frame = feed.read()
	frame =cv.flip(frame, 1)
	gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

	faces = detector(gray, 0)

	face_rectangle = face_cascade.detectMultiScale(gray, 1.5, 5)

	for (x, y, w, h) in face_rectangle:
		cv.rectangle(frame, (x,y), (x+w, y+h), (255, 0, 0), 2)

	for face in faces:

		shape = predictor(gray, face)
		shape = face_utils.shape_to_np(shape)

		left_eye = shape[lStart:lEnd]
		right_eye = shape[rStart:rEnd]

		leftEyeAspectRatio = eye_aspect_Ratio(left_eye)
		rightEyeAspectRatio = eye_aspect_Ratio(right_eye)

		eyeAspectRatio = (leftEyeAspectRatio + rightEyeAspectRatio) / 2


		leftEyeHull = cv.convexHull(left_eye)
		rightEyeHull = cv.convexHull(right_eye)

		cv.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
		cv.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)

		if(eyeAspectRatio < EYE_AR_THRESHOLD):
			COUNTER += 1
			if COUNTER >= EYE_AR_CONSECUTIVE_FRAMES:
				if not Alarm:
					Alarm = True
					playsound('alarm.wav')
				cv.putText(frame, 'DROWSINESS ALERT!', (10, 30),
				 cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
		else:
			COUNTER = 0
			Alarm = False

	cv.imshow('Video', frame)
	if(cv.waitKey(1) & 0xFF == ord('1')):
		break

feed.release()
cv.destroyAllWindows()

