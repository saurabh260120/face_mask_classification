import tensorflow as tf
import streamlit as st
from streamlit_webrtc import webrtc_streamer, RTCConfiguration
import av
import cv2
import numpy as np

def load_model():
  model=tf.keras.models.load_model('face_detection.h5')
  return model
with st.spinner('Model is being loaded..'):
  model=load_model()

st.write("""# Face Mask Classification""")

cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

def predict(image):
	image = image.to_ndarray(format="bgr24")
	image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
	input_image_resized = cv2.resize(image, (128,128))
	print(input_image_resized.shape)
	input_image_scaled = input_image_resized/255
	input_image_reshaped = np.reshape(input_image_scaled, [1,128,128,3])
	input_prediction = model.predict(input_image_reshaped)
	print(input_prediction)
	pred_label = input_prediction[0][0]
	
	if pred_label>=0.5:
		return "With Mask"
	else :
		return "Without_Mask"
	

class VideoProcessor:
	def recv(self, frame):
		frm = frame.to_ndarray(format="bgr24")

		faces = cascade.detectMultiScale(cv2.cvtColor(frm, cv2.COLOR_BGR2GRAY), 1.1, 3)

		for x,y,w,h in faces:
			cv2.rectangle(frm, (x,y), (x+w, y+h), (0,255,0), 3)
			texttoshow="Loading"
			try :
				texttoshow=predict(frame)
				print(texttoshow)
			except:
				print("model Didn't loaded....")
			new_image = cv2.putText(frm,text = texttoshow,org = (x, y),fontFace = cv2.FONT_HERSHEY_DUPLEX,fontScale = 1.0,color = (125, 246, 55),thickness = 3)
		return av.VideoFrame.from_ndarray(frm, format='bgr24')

webrtc_streamer(key="key", video_processor_factory=VideoProcessor,rtc_configuration=RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}))