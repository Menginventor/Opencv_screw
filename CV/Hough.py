import cv2
import numpy as np
from matplotlib import pyplot as plt
img = cv2.imread('screw1.jpg')

gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)


blur  =  cv2.blur(gray,(25,25))
median = cv2.medianBlur(gray, 25)
edges = cv2.Canny(median,0,255)
lines = cv2.HoughLines(edges,1,np.pi/180,200)
for line in lines:
    for rho,theta in line:
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a*rho
        y0 = b*rho
        x1 = int(x0 + 1000*(-b))
        y1 = int(y0 + 1000*(a))
        x2 = int(x0 - 1000*(-b))
        y2 = int(y0 - 1000*(a))

    cv2.line(img,(x1,y1),(x2,y2),(0,0,255),2)


plt.subplot(221),plt.imshow(median, cmap='gray'),plt.title('median')
plt.xticks([]), plt.yticks([])


plt.subplot(222),plt.imshow(edges, cmap='gray'),plt.title('Edges')
plt.xticks([]), plt.yticks([])
plt.show()