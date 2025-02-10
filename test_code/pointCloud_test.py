import cv2
import pcl
import numpy as np

# Load the image
img = cv2.imread('/home/downina3/Downloads/0027.JPG')

# Convert the image to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Detect keypoints using SIFT
sift = cv2.SIFT_create()
kp = sift.detect(gray, None)

# Compute the descriptors
kp, des = sift.compute(gray, kp)

# Create a point cloud from the keypoints
pc = pcl.PointCloud()
pc.from_array(np.array([kp.pt for kp in kp]))

# Visualize the point cloud
pcl.visualization.CloudViewing.showCloud(pc)
