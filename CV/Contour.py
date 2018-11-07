import cv2

import numpy as np
from matplotlib import pyplot as plt
img = cv2.imread("screw1.jpg")
#plt.subplot(441),plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)),plt.title('Original ')
#plt.xticks([]), plt.yticks([])
#
#gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
gray = cv2.convertScaleAbs(img)

#plt.subplot(442),plt.imshow(gray, cmap='gray'),plt.title('Gray ')
plt.xticks([]), plt.yticks([])

inv = cv2.bitwise_not(gray)
#plt.subplot(443),plt.imshow(inv, cmap='gray'),plt.title('Invert ')
#plt.xticks([]), plt.yticks([])

blur  =  cv2.blur(inv,(5,5))
#plt.subplot(444),plt.imshow(blur, cmap='gray'),plt.title('Blur ')
#plt.xticks([]), plt.yticks([])


edged = cv2.Canny(inv,25, 255)
edged = cv2.dilate(edged, None, iterations=5)
edged = cv2.erode(edged, None, iterations=5)


#plt.subplot(445),plt.imshow(edged, cmap='gray'),plt.title('Blur ')
#plt.xticks([]), plt.yticks([])

ret,thresh = cv2.threshold(edged,127,255,0)
cnts = im2, contours, hierarchy = cv2.findContours(thresh,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
# loop over the contours individually
orig = img.copy()
cv2.drawContours(orig, contours, -1, (0,255,0), 3)

plt.subplot(221),plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)),plt.title('Original')
plt.xticks([]), plt.yticks([])


plt.subplot(222),plt.imshow(edged,cmap='gray'),plt.title('Edged')
plt.xticks([]), plt.yticks([])

plt.subplot(223),plt.imshow(cv2.cvtColor(orig, cv2.COLOR_BGR2RGB)),plt.title('Contour')
plt.xticks([]), plt.yticks([])
orig2 = img.copy()

# create hull array for convex hull points
hull_arr = []
simple = []
# calculate points for each contour
for i in range(len(contours)):
    # creating convex hull object for each contour
    hull = cv2.convexHull(contours[i], True)
    hull_arr.append(hull)
    peri = cv2.arcLength(contours[i], True)
    approx = cv2.approxPolyDP(contours[i], 0.01 * peri, True)
    simple.append(approx)
    #x, y, w, h = cv2.boundingRect(contours[i])
    #cv2.rectangle(orig2, (x, y), (x + w, y + h), (0, 0, 255), 2)
    rows, cols = orig2.shape[:2]
    [vx, vy, x, y] = cv2.fitLine(contours[i], cv2.DIST_L2, 0, 0.01, 0.01)
    lefty = int((-x * vy / vx) + y)
    righty = int(((cols - x) * vy / vx) + y)
    cv2.line(orig2, (cols - 1, righty), (0, lefty), (0, 0, 255), 3)

    M = cv2.moments(hull)
    cX = int(M['m10'] / M['m00'])
    cY = int(M['m01'] / M['m00'])
    cv2.circle(orig2, (cX, cY), 25, (255, 0, 255), -1)


cv2.drawContours(orig2, hull_arr, -1, (0, 255, 0), 3)
cv2.drawContours(orig2, simple, -1, (255, 0, 0), 3)
plt.subplot(224),plt.imshow(cv2.cvtColor(orig2, cv2.COLOR_BGR2RGB)),plt.title('Contour')
plt.xticks([]), plt.yticks([])
plt.show()