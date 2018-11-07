import cv2
import numpy as np
from matplotlib import pyplot as plt
import os
def crop(src,x,y,w,h,offset = 0):
    height = np.size(src, 0)
    width = np.size(src, 1)
    crop_ymin = y - offset
    if crop_ymin < 0:
        crop_ymin = 0
    crop_xmin = x - offset
    if crop_xmin < 0:
        crop_xmin = 0
    crop_ymax = y + h + offset
    if crop_ymax >= height:
        crop_ymax = height - 1
    crop_xmax = x + w + offset
    if crop_xmax >= width:
        crop_xmax = width - 1

    return (src[crop_ymin:crop_ymax, crop_xmin:crop_xmax],)
def line_sort(Lines):#sorting line by length

    length = [np.linalg.norm(np.subtract(l[0][0:2],l[0][2:4])) for l in Lines]
    sorted_line_index = [x for _, x in sorted(zip(length,range(len(Lines))))]
    #print([Lines[i] for i in sorted_line_index])

    #print(np.sort(zipped.view('i8,i8,i8'), order=['f1'], axis=0).view(np.int))
    return [Lines[i] for i in sorted_line_index]

def dist_line_to_point(Line,Point):
    #print(Line[0])
    x1,y1,x2,y2 = Line[0]
    x0,y0 = Point
    return abs((y2-y1)*x0-(x2-x1)*y0+x2*y1-y2*x1)/np.linalg.norm(np.subtract(Line[0][0:2],Line[0][2:4]))



img = cv2.imread('screw1.jpg')

gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)


blur  =  cv2.blur(gray,(25,25))
median = cv2.medianBlur(gray, 25)
edges = cv2.Canny(median,75,255)
edges = cv2.dilate(edges, None, iterations=1)
#edges = cv2.erode(edges, None, iterations=3)


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
result_contours = []
for i in range(len(contours)):
    cnt = contours[i]

    x,y,w,h = cv2.boundingRect(cnt)
    cv2.rectangle(img_cnts,(x,y),(x+w,y+h),(0,0,255),2)
    ####
    offset = 50
    height = np.size(edges, 0)
    width = np.size(edges, 1)
    crop_ymin = y - offset
    if crop_ymin < 0:
        crop_ymin = 0
    crop_xmin = x - offset
    if crop_xmin < 0:
        crop_xmin = 0
    crop_ymax = y + h + offset
    if crop_ymax >= height:
        crop_ymax = height - 1
    crop_xmax = x + w + offset
    if crop_xmax >= width:
        crop_xmax = width - 1
    crop_img =edges[crop_ymin:crop_ymax, crop_xmin:crop_xmax]
    #crop_img = crop(edges,x,y,w,h, 50)


    Plines = cv2.HoughLinesP(crop_img, rho = 1, theta = 0.5 * np.pi / 180, threshold=200, minLineLength=25, maxLineGap=50)
    Hough  = cv2.cvtColor(crop_img, cv2.COLOR_GRAY2RGB)
    #print(len(Plines))
    #Plines = remove_same_line(Plines)
    #print(len(Plines))
    #print(Plines)
    Lines_sorted  = line_sort(Plines)[::-1]
    Line_1 = Lines_sorted[0]
    for Pline in Lines_sorted[1::]:
        d = dist_line_to_point(Line_1,Pline[0][0:2])
        if d > 10:
            Line_2 = Pline
            break
    x1, y1,x2, y2 = Line_1[0]
    cv2.line(Hough, (x1, y1), (x2, y2), (0, 255, 0), 2)
    x3, y3, x4, y4 = Line_2[0]
    cv2.line(Hough, (x3, y3), (x4, y4), (0, 255, 0), 2)
    screw_contour = np.array([(x1, y1), (x2, y2), (x4, y4), (x3, y3)], dtype=np.int)
    print(cv2.isContourConvex(screw_contour))
    #cv2.drawContours(Hough, [screw_contour], -1, (0, 0, 255), 3)
    rect = cv2.minAreaRect(screw_contour)
    box = cv2.boxPoints(rect)
    box = np.int0(box)
    cv2.drawContours(Hough, [box], 0, (255, 0, 0), 2)
    box_offset = box.copy()
    for p in box_offset:
        p[0] += crop_xmin
        p[1] += crop_ymin

    #    print(i,'Pline',Pline)

        #print(line_sort(Plines))

    cv2.imwrite(os.getcwd() + '/Crop_img/crop-' + str(i) + '.jpg', Hough)

    result_contours.append(box_offset)
cv2.drawContours(img_cnts, result_contours, -1, (255,0,0), 3)
plt.subplot(111),plt.imshow(cv2.cvtColor(img_cnts, cv2.COLOR_BGR2RGB)),plt.title('Contours')
plt.xticks([]), plt.yticks([])
plt.xticks([]), plt.yticks([])
plt.show()