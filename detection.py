import numpy as np
from cv2 import cv2
import matplotlib.pyplot as plt



def canny(image):
    gray = cv2.cvtColor(image,cv2.COLOR_RGB2GRAY)
    blured = cv2.GaussianBlur(gray,(5,5),0)
    canny = cv2.Canny(blured,50,150)
    return canny


def create_coordinates(image, line_parameters): 
    slope, intercept = line_parameters 
    y1 = image.shape[0] 
    y2 = int(y1 * (3 / 5)) 
    x1 = int((y1 - intercept) / slope) 
    x2 = int((y2 - intercept) / slope) 
    return np.array([x1, y1, x2, y2]) 


def average_slope_intercept(image, lines): 
    left_fit = [] 
    right_fit = [] 
    for line in lines: 
        x1, y1, x2, y2 = line.reshape(4)  
        parameters = np.polyfit((x1, x2), (y1, y2), 1)  
        slope = parameters[0] 
        intercept = parameters[1] 
        if slope < 0: 
            left_fit.append((slope, intercept)) 
        else: 
            right_fit.append((slope, intercept)) 
    left_fit_average = np.average(left_fit, axis = 0) 
    right_fit_average = np.average(right_fit, axis = 0) 
    left_line = create_coordinates(image, left_fit_average) 
    right_line = create_coordinates(image, right_fit_average) 
    return np.array([left_line, right_line])

            
def displayLines(image, lines):
    line_image = np.zeros_like(image)
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line.reshape(4)
            cv2.line(line_image, (x1, y1), (x2, y2), (255, 0, 0), 10)
    return line_image



def regionOfStudy(image): 
    height = image.shape[0] 
    polygons = np.array([ 
        [(200, height), (1100, height), (550, 250)] 
        ]) 
    mask = np.zeros_like(image) # a mask of same Dimensions filled with zeros 
    cv2.fillPoly(mask, polygons, 255) # 255 indicates white color
    masked_image = cv2.bitwise_and(image, mask)  
    return masked_image





cap = cv2.VideoCapture("video.mp4")
while(cap.isOpened()):
    _, frame = cap.read()
    cannys = canny(frame)
    cropped = regionOfStudy(cannys)
    lines =  cv2.HoughLinesP(cropped, 2, np.pi/180, 100, np.array([]), minLineLength=40, maxLineGap=5)
    averagedLine = average_slope_intercept(frame,lines)
    line_image = displayLines(frame,averagedLine)
    mixedImage = cv2.addWeighted(frame, 0.8, line_image, 1, 1)
    cv2.imshow('Lane Detection',mixedImage)
    cv2.waitKey(1)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
  
cap.release()  
cv2.destroyAllWindows()  