import cv2
import numpy as np
from matplotlib import pyplot as plt
img = cv2.imread('screw1.jpg')

gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)


blur  =  cv2.blur(gray,(25,25))
median = cv2.medianBlur(gray, 25)
edges = cv2.Canny(median,75,255)
edges = cv2.dilate(edges, None, iterations=5)
edges = cv2.erode(edges, None, iterations=5)
lines = cv2.HoughLines(edges,1,np.pi/180,100)
#Plines
minLineLength = 100
maxLineGap = 10
Plines = cv2.HoughLinesP(edges,rho = 1,theta = 1*np.pi/180,threshold = 50,minLineLength = 100,maxLineGap = 100)
#Plines
hough = backtorgb = cv2.cvtColor(edges,cv2.COLOR_GRAY2RGB)
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
        #cv2.line(hough,(x1,y1),(x2,y2),(0,0,255),2)
for Pline in Plines:
    for x1,y1,x2,y2 in Pline:
        cv2.line(hough,(x1,y1),(x2,y2),(0,255,0),2)
        pass

plt.subplot(221),plt.imshow(median, cmap='gray'),plt.title('median')
plt.xticks([]), plt.yticks([])


plt.subplot(222),plt.imshow(edges, cmap='gray'),plt.title('Edges')
plt.xticks([]), plt.yticks([])


#plt.subplot(223),plt.imshow(cv2.cvtColor(hough, cv2.COLOR_BGR2RGB)),plt.title('hough')
plt.subplot(223),plt.imshow(hough, cmap='gray'),plt.title('Hough')
plt.xticks([]), plt.yticks([])
plt.show()