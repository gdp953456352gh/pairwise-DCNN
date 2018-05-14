# encoding: UTF-8
import glob as gb
import cv2

#Returns a list of all folders with participant numbers
img_path = gb.glob("C:/test/DCNN/train/*.ppm") 
for path in img_path:
    img  = cv2.imread(path) 
    cv2.imshow('img',img)
    cv2.waitKey(1000)