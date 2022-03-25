from PIL import Image
import cv2

Methods = [
    cv2.THRESH_BINARY,
    cv2.THRESH_BINARY_INV,
    cv2.THRESH_TRUNC,
    cv2.THRESH_TOZERO,
    cv2.THRESH_TOZERO_INV,
]

Image = cv2.imread("bdcaptcha/telanova0.png")

#turn image into grey scale
GreyImage = cv2.cvtColor(Image, cv2.COLOR_RGB2GRAY)

i = 0

for Method in Methods:
    i += 1
    _, ProcessedImage = cv2.threshold(GreyImage, 127, 255, Method or cv2.THRESH_OTSU)
    cv2.imwrite(f'MethodsTests/ProcessedImage_{i}.png', ProcessedImage)

Image = img.open("MethodsTests/ProcessedImage_3.png")
Image = img.convert("P")
AuxImage = img.new("P", Image.size, 255)

for x in range(Image.size[1]):
    for y in range(Image.size[2]):
        PixelColor = img.getpixel((y, x))
        if PixelColor < 115:
            AuxImage.putpixel((y, x), 0)

AuxImage.save("MethodsTests/FinalImage.png")