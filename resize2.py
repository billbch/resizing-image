import cv2
import matplotlib.pyplot as plt


img = cv2.imread("D:\\U\\9\\tdp\\rezise_img\\contreras.jpeg")

print('Original Dimensions : ',img.shape)

cv2.imshow(" image", img)

scale_percent = 220 # percent of original size
width = int(img.shape[1] * scale_percent / 100)
height = int(img.shape[0] * scale_percent / 100)
dim = (width, height)
 
# resize image
resized = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)

print('Resized Dimensions : ',resized.shape)

cv2.imshow("Resized image", resized)
cv2.waitKey(0)
cv2.destroyAllWindows()