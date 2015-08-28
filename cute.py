# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv


def main():
  kitty = cv.imread("kitty.png")
  pattern = cv.imread("pattern.png")

  cap = cv.VideoCapture(1)
  print kitty.shape
  kitty_resized = cv.resize(kitty,(640,480))
  orb = cv.ORB()
  keypoints, descriptors = orb.detectAndCompute(pattern, None)
  pic = pattern
  cv.drawKeypoints(pattern,keypoints,pic)
  cv.imshow('pic',pic)
  while(True):
    # Capture frame-by-frame
    _, frame = cap.read()
    
    # Our operations on the frame come here
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    graypic = gray
    #graypic = gray
    orb = cv.ORB()
    cam_keypoints, cam_descriptors = orb.detectAndCompute(gray, None)
    cv.drawKeypoints(gray,cam_keypoints,graypic)
    print(graypic)
    cv.imshow('graypic',graypic)
    #print gray.shape
    # Display the resulting frame
    cv.imshow('frame',gray)
    half = (kitty_resized + frame)/2
    cv.imshow('half', half)
    if cv.waitKey(1) & 0xFF == ord('q'):
      break

  # When everything done, release the capture
  cap.release()
  cv.destroyAllWindows()


if __name__ == '__main__':
  main()
