# /usr/bin/python3
import cv2
import numpy as np
import sys
import tensorflow as tf

from model import predict, image_to_tensor, deepnn

CASC_PATH = './data/haarcascade_files/haarcascade_frontalface_default.xml'
cascade_classifier = cv2.CascadeClassifier(CASC_PATH)
EMOTIONS = ['angry', 'disgusted', 'fearful', 'happy', 'sad', 'surprised', 'neutral']

def format_image(image):
  if len(image.shape) > 2 and image.shape[2] == 3:
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
  faces = cascade_classifier.detectMultiScale(
    image,
    scaleFactor = 1.3,
    minNeighbors = 5
  )
  # None is no face found in image
  if not len(faces) > 0:
    return None, None
  max_are_face = faces[0]
  for face in faces:
    if face[2] * face[3] > max_are_face[2] * max_are_face[3]:
      max_are_face = face
  # face to image
  face_coor =  max_are_face
  image = image[face_coor[1]:(face_coor[1] + face_coor[2]), face_coor[0]:(face_coor[0] + face_coor[3])]
  # Resize image to network size
  try:
    image = cv2.resize(image, (48, 48), interpolation=cv2.INTER_CUBIC)
  except Exception:
    print("[+} Problem during resize")
    return None, None
  return  image, face_coor

def face_dect(image):
  """
  Detecting faces in image
  :param image: 
  :return:  the coordinate of max face
  """
  if len(image.shape) > 2 and image.shape[2] == 3:
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
  faces = cascade_classifier.detectMultiScale(
    image,
    scaleFactor = 1.3,
    minNeighbors = 5
  )
  if not len(faces) > 0:
    return None
  max_face = faces[0]
  for face in faces:
    if face[2] * face[3] > max_face[2] * max_face[3]:
      max_face = face
  face_image = image[max_face[1]:(max_face[1] + max_face[2]), max_face[0]:(max_face[0] + max_face[3])]
  try:
    image = cv2.resize(face_image, (48, 48), interpolation=cv2.INTER_CUBIC) / 255.
  except Exception:
    print("[+} Problem during resize")
    return None
  return face_image

def resize_image(image, size):
  try:
    image = cv2.resize(image, size, interpolation=cv2.INTER_CUBIC) / 255.
  except Exception:
    print("+} Problem during resize")
    return None
  return image

def draw_emotion():
  pass

def demo(modelPath, showBox=False):
  face_x = tf.placeholder(tf.float32, [None, 2304])
  y_conv = deepnn(face_x)
  probs = tf.nn.softmax(y_conv)

  saver = tf.train.Saver()
  ckpt = tf.train.get_checkpoint_state(modelPath)
  sess = tf.Session()
  if ckpt and ckpt.model_checkpoint_path:
    saver.restore(sess, ckpt.model_checkpoint_path)
    print('Restore model sucsses!!\nNOTE: Press SPACE on keyboard to capture face.')

  feelings_faces = []
  for index, emotion in enumerate(EMOTIONS):
    feelings_faces.append(cv2.imread('./data/emojis/' + emotion + '.png', -1))
  video_captor = cv2.VideoCapture(0)

  emoji_face = []
  result = None

  while True:
    ret, frame = video_captor.read()
    detected_face, face_coor = format_image(frame)
    if showBox:
      if face_coor is not None:
        [x,y,w,h] = face_coor
        cv2.rectangle(frame, (x,y), (x+w,y+h), (255,0,0), 2)

    if cv2.waitKey(1) & 0xFF == ord(' '):

      if detected_face is not None:
        cv2.imwrite('a.jpg', detected_face)
        tensor = image_to_tensor(detected_face)
        result = sess.run(probs, feed_dict={face_x: tensor})
        # print(result)
    if result is not None:
      for index, emotion in enumerate(EMOTIONS):
        cv2.putText(frame, emotion, (10, index * 20 + 20), cv2.FONT_HERSHEY_PLAIN, 0.5, (0, 255, 0), 1)
        cv2.rectangle(frame, (130, index * 20 + 10), (130 + int(result[0][index] * 100), (index + 1) * 20 + 4),
                      (255, 0, 0), -1)
        emoji_face = feelings_faces[np.argmax(result[0])]

      for c in range(0, 3):
        frame[200:320, 10:130, c] = emoji_face[:, :, c] * (emoji_face[:, :, 3] / 255.0) + frame[200:320, 10:130, c] * (1.0 - emoji_face[:, :, 3] / 255.0)
    cv2.imshow('face', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
      break

