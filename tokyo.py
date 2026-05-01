import cv2 as cv

img = cv.imread("PedroH.jpeg")

gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

cv.imshow("Display Window", gray)
k = cv.waitKey(0)