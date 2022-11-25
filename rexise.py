import cv2
import time
import matplotlib.pyplot as plt

# Cropout OpenCV logo
img = cv2.imread("D:\\U\\9\\tdp\\rezise_img\\contreras.jpeg")
cv2.imshow(" image", img)
print(img.shape)
sr = cv2.dnn_superres.DnnSuperResImpl_create()
 
path = "EDSR_x4.pb"
 
sr.readModel(path)
 
sr.setModel("edsr",4)
 
result = sr.upsample(img)

sr2 = cv2.dnn_superres.DnnSuperResImpl_create()
 
path2 = "LapSRN_x4.pb"

sr2.readModel(path2)
 
sr2.setModel("lapsrn",4)
 
result2 = sr2.upsample(img)

sr3 = cv2.dnn_superres.DnnSuperResImpl_create()
 
path3 = "FSRCNN_x4.pb"
 
sr3.readModel(path3)
 
sr3.setModel("fsrcnn",4)
 
result3 = sr3.upsample(img)
 
# Resized image
resized = cv2.resize(img,dsize=None,fx=4,fy=4)
 
cv2.imshow(" image", img)
cv2.imshow(" result", result)
cv2.imshow(" result2", result2)
cv2.imshow(" result3", result3)
#cv2.imshow(" resized", resized)

cv2.waitKey(0)
cv2.destroyAllWindows()