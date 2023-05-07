from mtcnn.mtcnn import MTCNN
import cv2
import numpy as np

from keras_vggface.utils import preprocess_input
from keras_vggface.vggface import VGGFace
from scipy.spatial.distance import cosine

import os
isExist = os.path.exists("faces")
if not isExist:
  os.makedirs("faces")
isExist = os.path.exists("np")
if not isExist:
  os.makedirs("np")

required_size=(224, 224)
threshold = 0.4
font = cv2.FONT_HERSHEY_PLAIN

ids = []
embedding = []
with open('embedding.txt', 'r') as the_file:
    ll = the_file.readlines()

for l in ll:
    spli = l.split(';')
    ids.append(spli[1])
    embedding.append(np.load(spli[3].strip()))

print(ll)

model = VGGFace(model='resnet50',
                include_top=False,
                input_shape=(224, 224, 3),
                pooling='avg')

detector = MTCNN()
vid = cv2.VideoCapture(0)

def get_model_scores(faces):
    samples = np.asarray([faces], 'float32')

    # prepare the data for the model
    samples = preprocess_input(samples, version=2)

    # create a vggface model object

    if samples.shape[1] == 1:
        samples = np.squeeze(samples, axis=1)
    # perform prediction
    return model.predict(samples)

def highlight_faces(image, faces):
    image = np.copy(image)
    color = (255, 0, 0)
    thickness = 1
    for face in faces:
        x, y, width, height = face['box']
        image = cv2.rectangle(image, (x, y), (x + width, y + height), color, thickness)

        xx = frame[y:(y + height), x:(x + width), :]
        xx = cv2.resize(xx, required_size)
        score = get_model_scores(xx)

        min_emb_score = 2
        id = -1
        id_help = 0
        for emb in embedding:
            emb_score = cosine(score, emb)
            if emb_score < min_emb_score and emb_score < threshold:
                min_emb_score = emb_score
                id = id_help
            id_help = id_help + 1

        if id != -1:
            my_str = str(ids[id])
            cv2.putText(image, my_str, (x + width, y), font, 2, (0, 0, 255))
            my_str = "score:" + "{:.3f}".format(min_emb_score)
            cv2.putText(image, my_str, (x + width, y + 15), font, 2, (0, 0, 255))
    cv2.imshow("faces", image)

while (True):
    # Capture the video frame
    # by frame
    ret, frame = vid.read()

    faces = detector.detect_faces(frame)
    highlight_faces(frame, faces)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# After the loop release the cap object
vid.release()
# Destroy all the windows
cv2.destroyAllWindows()
