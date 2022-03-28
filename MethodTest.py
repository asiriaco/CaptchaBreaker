import pil.Image as Image
import cv2

Methods = [
    cv2.THRESH_BINARY,
    cv2.THRESH_BINARY_INV,
    cv2.THRESH_TRUNC,
    cv2.THRESH_TOZERO,
    cv2.THRESH_TOZERO_INV,
]

image = cv2.imread("bdcaptcha/telanova2.png")

#turn image into grey scale
GreyImage = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

i = 0

for Method in Methods:
    i += 1
    _, ProcessedImage = cv2.threshold(GreyImage, 127, 255, Method or cv2.THRESH_OTSU)
    cv2.imwrite(f'MethodsTests/ProcessedImage_{i}.png', ProcessedImage)

img = Image.open("MethodsTests/ProcessedImage_3.png")
img = img.convert('L')
AuxImage = Image.new('L', img.size, 255)


for x in range(img.size[1]):
    for y in range(img.size[0]):
        coordinate = y, x
        PixelColor = img.getpixel(coordinate)
        if PixelColor < 115:
            AuxImage.putpixel((y, x), 0)
AuxImage.save("MethodsTests/FinalImage.png")