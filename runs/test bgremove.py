import cv2

img1 = cv2.imread('C:\\Tensorflow\\Dataset\\bgremove\\1.jpg')
img2 = cv2.imread('C:\\Tensorflow\\Dataset\\bgremove\\5.jpg')

height, width = img1.shape[:2]
img2 = cv2.resize(img2, (width, height))
gimg1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
gimg2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
subtracted = cv2.subtract(gimg2, gimg1)
subtracted = cv2.resize(subtracted, (1920, 1080))
cv2.imshow('image', subtracted)

# To close the window
cv2.waitKey(0)
cv2.destroyAllWindows()