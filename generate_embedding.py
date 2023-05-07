import cv2
from keras_vggface.utils import preprocess_input
from keras_vggface.vggface import VGGFace
import numpy as np
import os
import glob

isExist = os.path.exists("faces")
if not isExist:
  os.makedirs("faces")
isExist = os.path.exists("np")
if not isExist:
  os.makedirs("np")

required_size=(224, 224)
files = glob.glob("faces/*")
files_count = len(files)

model = VGGFace(model='resnet50',
                include_top=False,
                input_shape=(224, 224, 3),
                pooling='avg')


def get_model_scores(faces):
  samples = np.asarray([faces], 'float32')

  # prepare the data for the model
  samples = preprocess_input(samples, version=2)

  # create a vggface model object

  if samples.shape[1] == 1:
    samples = np.squeeze(samples, axis=1)
  # perform prediction
  return model.predict(samples)

with open('embedding.txt', 'w') as the_file:
    for f in files:
        ff = cv2.imread(f)
        score = get_model_scores(ff)
        np_file = "np/" + os.path.basename(f)
        id = os.path.basename(f).split('.')
        print(np_file)
        np.save(np_file, score)
        the_file.write(id[0] + ";" + id[0] + ";" + f + ";" + np_file + ".npy\n")
