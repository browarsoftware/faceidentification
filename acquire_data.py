from mtcnn.mtcnn import MTCNN
import cv2
import numpy as np

import glob
import os
isExist = os.path.exists("faces")
if not isExist:
  os.makedirs("faces")
isExist = os.path.exists("np")
if not isExist:
  os.makedirs("np")

required_size=(224, 224)
files = glob.glob("faces/*")
files_count = len(files)

detector = MTCNN()
vid = cv2.VideoCapture(0)

def highlight_faces(image, faces):
  image = np.copy(image)
  color = (255, 0, 0)
  thickness = 1
  for face in faces:
    x, y, width, height = face['box']
    image = cv2.rectangle(image, (x, y), (x + width, y + height), color, thickness)
  return image

while (True):
  # Capture the video frame
  # by frame
  ret, frame = vid.read()

  faces = detector.detect_faces(frame)
  image = highlight_faces(frame, faces)

  # Display the resulting frame
  cv2.imshow('frame', image)
  # the 'q' button is set as the
  # quitting button you may use any
  # desired button of your choice
  if cv2.waitKey(1) & 0xFF == ord('s'):
    for face in faces:
      x, y, width, height = face['box']
      xx = frame[y:(y + height), x:(x + width), :]
      file_name = 'faces/' + str(files_count) + ".png"
      print(file_name)
      xx = cv2.resize(xx, required_size)

      cv2.imwrite(file_name, xx)

      files_count = files_count + 1

  if cv2.waitKey(1) & 0xFF == ord('q'):
    break

# After the loop release the cap object
vid.release()
# Destroy all the windows
cv2.destroyAllWindows()
