import cv2
import numpy as np
from matplotlib import pyplot as plt
import os
img = cv2.imread('screw1.jpg')

gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)


blur  =  cv2.blur(gray,(25,25))
median = cv2.medianBlur(gray, 25)
edges = cv2.Canny(median,75,255)
edges = cv2.dilate(edges, None, iterations=5)
edges = cv2.erode(edges, None, iterations=5)


ret,thresh = cv2.threshold(edges,127,255,0)
cnts = im2, contours, hierarchy = cv2.findContours(thresh,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)


'''
minLineLength = 100
maxLineGap = 10
Plines = cv2.HoughLinesP(edges,rho = 1,theta = 1*np.pi/180,threshold = 50,minLineLength = 100,maxLineGap = 100)
#Plines
hough = backtorgb = cv2.cvtColor(edges,cv2.COLOR_GRAY2RGB)

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
'''
img_cnts = img.copy()

cv2.drawContours(img_cnts, contours, -1, (0,255,0), 3)
folder = os.getcwd()+'/Crop_img'
for the_file in os.listdir(folder):
    file_path = os.path.join(folder, the_file)
    try:
        if os.path.isfile(file_path):
            os.unlink(file_path)
        # elif os.path.isdir(file_path): shutil.rmtree(file_path)
    except Exception as e:
        print(e)
for i in range(len(contours)):
    cnt = contours[i]
    x,y,w,h = cv2.boundingRect(cnt)
    cv2.rectangle(img_cnts,(x,y),(x+w,y+h),(0,0,255),2)
    ####

    height = np.size(edges, 0)
    width = np.size(edges, 1)
    offset = 50
    crop_ymin = y - offset
    if crop_ymin<0:
        crop_ymin = 0
    crop_xmin = x - offset
    if crop_xmin < 0:
        crop_xmin = 0

    crop_ymax = y + h + offset
    if crop_ymax >= height:
        crop_ymax = height-1
    crop_xmax = x + w + offset
    if crop_xmax >= width:
        crop_xmax = width-1

    crop_img = edges[crop_ymin:crop_ymax, crop_xmin:crop_xmax]
    cv2.imwrite(os.getcwd()+'/Crop_img/crop-'+str(i)+'.jpg',crop_img)



plt.subplot(221),plt.imshow(cv2.cvtColor(img_cnts, cv2.COLOR_BGR2RGB)),plt.title('Contours')
plt.xticks([]), plt.yticks([])
plt.xticks([]), plt.yticks([])
plt.show()