from keras.models import load_model
from helpers import resize_to_fit
from imutils import paths
import numpy as np
import pandas as pd
import cv2
import pickle
from CaptchaProcessor import process


def BreakCaptcha():

    Captchas = []

    with open("labelModels.dat", "rb") as Translator:
        lb = pickle.load(Translator)

    model = load_model("TrainedModel.hdf5")
    process("Solve", target_dir="Solve")
     #####
    files = list(paths.list_images("Solve"))

    for file in files:
        img = cv2.imread(file)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        _, newimg = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV)

        edges, _ = cv2.findContours(newimg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        regions = []

        for edge in edges:
            (x, y, width, height) = cv2.boundingRect(edge)
            area = cv2.contourArea(edge)
            if area > 115:
                regions.append((x, y, width, height))

        regions = sorted(regions, key=lambda x:x[0])
        FinalImage = cv2.merge([img] * 3)
        forecast = []
        i = 0

        for rectangle in regions:

            x, y, width, height = rectangle
            LetterImage = img[y - 2:y + height + 2, x - 2:x + width + 2]
            LetterImage = resize_to_fit(LetterImage, 20, 20)

            LetterImage = np.expand_dims(LetterImage, axis=2)
            LetterImage = np.expand_dims(LetterImage, axis=0)

            ExpectedChar = model.predict(LetterImage)
            ExpectedChar = lb.inverse_transform(ExpectedChar)[0]

            forecast.append(ExpectedChar)

            cv2.rectangle(FinalImage, (x - 2, y - 2), (x + width + 2, y + height + 2), (0, 0, 255), 1)

        expectedCaptcha = "".join(forecast)
        Captchas.append(expectedCaptcha)
       # print(expectedCaptcha)
    return Captchas

if __name__ == "__main__":

    captchas = BreakCaptcha()
    df = pd.DataFrame(captchas)
    print (df)
    df.columns = ["Captcha"]
    df.to_csv("Captchas.csv", index=False)