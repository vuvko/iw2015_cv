# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv
from skimage import transform as tf
from sklearn.cluster import KMeans


def find_first(l, el):
  for idx, cur in enumerate(l):
    if cur == el:
      return idx
  return None


def get_edges(image, orb):
  n_clusters = 250
  clust = KMeans(n_clusters=n_clusters, init='k-means++', max_iter=300, n_init=5, n_jobs=2)
  keypoints, descriptors = orb.detectAndCompute(image, None)
  clust.fit([k.pt for k in keypoints])
  labels = clust.labels_
  first_i = [find_first(labels, label) for label in xrange(n_clusters)]
  return [keypoints[i] for i in first_i], descriptors[first_i]


def get_transform(bf,descr0, key0, descr1, key1):
  if descr0 is None or descr1 is None or descr0.shape[1] != descr1.shape[1]:
    return None, None
  matches = bf.knnMatch(descr0, descr1, k=2)
  if len(matches) < 20 or len(matches[0]) != 2:
    return None, None
  matches = filter(lambda m: m[0].distance < 0.9 * m[1].distance, matches)
  good_matches = [m1 for m1,m2 in matches]
  if len(good_matches) < 10:
    return None, None
  dst_pts = np.array([key0[m.queryIdx].pt for m in good_matches], dtype='float32')
  src_pts = np.array([key1[m.trainIdx].pt for m in good_matches], dtype='float32')
  return cv.findHomography(src_pts, dst_pts, cv.RANSAC, 15)


def main():
  print 'To quit press "q"'
  
  kitty = cv.imread("kitty.png")
  pattern = cv.imread("pattern.png")
  pattern_gray = cv.cvtColor(pattern, cv.COLOR_BGR2GRAY)
  
  cap = cv.VideoCapture(0)
  #print kitty.shape
  kitty_resized = cv.resize(kitty,(640,480))
  
  bf=cv.BFMatcher(cv.NORM_HAMMING)
  orb = cv.ORB(nfeatures=1000, scaleFactor=1.6, nlevels=8, edgeThreshold=51, WTA_K=2, patchSize=51)
  keypoints, descriptors = get_edges(pattern_gray, orb)
  
  pic = pattern
  cv.drawKeypoints(pattern,keypoints,pic, color=(0,0,255))
  cv.imshow('pic',pic)
  #return
  while(True):
    # Capture frame-by-frame
    _, frame = cap.read()
    
    # Our operations on the frame come here
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    #graypic = gray
    graypic = frame + 0
    cam_keypoints, cam_descriptors = orb.detectAndCompute(gray, None)
    cv.drawKeypoints(frame,cam_keypoints,graypic, color=(0,0,255))
    #print cam_keypoints
    #print len(cam_keypoints)
    
    cv.imshow('graypic',graypic)
    
    M, mask = get_transform(bf,descriptors, keypoints, cam_descriptors, cam_keypoints )
    out = frame
    if M is not None:
      kitty_warped = tf.warp(kitty_resized,tf.ProjectiveTransform(M))
      non_black_pixels = np.where(np.sum(kitty_warped, axis=2) > 0)
      out[non_black_pixels[0], non_black_pixels[1], :] = kitty_warped[non_black_pixels[0], non_black_pixels[1], :] * 255
      #out = out + kitty_warped
    cv.imshow('half', out)
    if cv.waitKey(1) & 0xFF == ord('q'):
      break

  # When everything done, release the capture
  cap.release()
  cv.destroyAllWindows()


if __name__ == '__main__':
  main()
