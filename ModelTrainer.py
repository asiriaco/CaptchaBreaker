import cv2
import os
import numpy as np
import pickle
from imutils import paths
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers.core import Flatten, Dense
from helpers import resize_to_fit


data = []
labels = []

ImageDB = "LettersDB"
Images = paths.list_images(ImageDB)

for file in Images:
    label = file.split(os.path.sep)[-2]
    img = cv2.imread(file)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = resize_to_fit(img, 20, 20)

    img = np.expand_dims(img, axis=2)

    labels.append(label)
    data.append(img)

data = np.array(data, dtype="float")/255
labels = np.array(labels)


(Xtrain, Xtest, Ytrain, Ytest) = train_test_split(data, labels, test_size=0.25, random_state=0)

lb = LabelBinarizer().fit(Ytrain)
Ytrain = lb.transform(Ytrain)
Ytest = lb.transform(Ytest)

with open("labelModels.dat", 'wb') as PickleFile:
    pickle.dump(lb, PickleFile)


model = Sequential()

#1st layer
model.add(Conv2D(20, (5, 5), padding = "same", input_shape=(20, 20, 1), activation="relu"))
model.add(MaxPooling2D(pool_size=(2,2), strides=(2, 2)))

#2ndlayer
model.add(Conv2D(50, (5, 5), padding="same", activation="relu"))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))


#3rd layer
model.add(Flatten())
model.add(Dense(500, activation="relu"))

#output layer
model.add(Dense(26, activation="softmax"))

#compile all layers
model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

#train neural link
model.fit(Xtrain, Ytrain, validation_data=(Xtest, Ytest), batch_size=26, epochs=10, verbose=1)

#save model
model.save("TrainedModel.hdf5")
