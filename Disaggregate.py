import cv2
import os
import glob

files = glob.glob("ProcessedImages/*")

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

    if len(regions) != 5:
        continue

    FinalImage = cv2.merge([img] * 3)
    i=0
    for rectangle in regions:
        i+=1
        x, y, width, height = rectangle
        LetterImage = img[y-2:y+height+2, x-2:x+width+2]
        filename = os.path.basename(file).replace("png", f"letter{i}.png")
        cv2.imwrite(f'Letters/{filename}', LetterImage)
        cv2.rectangle(FinalImage, (x-2, y-2), (x+width+2, y+height+2), (0, 0, 255), 1)

    filename = os.path.basename(file)
    cv2.imwrite(f"Identified/{filename}", FinalImage)