import cv2
import os
import glob
import pil.Image as Image

def process(origin_dir,  target_dir = "ProcessedImages"):
    files = glob.glob(f"{origin_dir}/*")

    for file in files:

        image = cv2.imread(file)
        GreyImage = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

        _, ProcessedImage = cv2.threshold(GreyImage, 127, 255, cv2.THRESH_TRUNC or cv2.THRESH_OTSU)
        filename = os.path.basename(file)
        cv2.imwrite(f'{target_dir}/{filename}', ProcessedImage)

    files = glob.glob(f"{target_dir}/*")
    for file in files:
        img = Image.open(file)
        img = img.convert('L')
        AuxImage = Image.new('L', img.size, 255)

        for x in range(img.size[1]):
            for y in range(img.size[0]):
                coordinate = y, x
                PixelColor = img.getpixel(coordinate)
                if PixelColor < 115:
                    AuxImage.putpixel((y, x), 0)
        filename = os.path.basename(file)
        AuxImage.save(f"{target_dir}/{filename}")


if __name__ == "__main__":
    process('bdcaptcha')