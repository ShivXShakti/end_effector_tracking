"""#!/usr/bin/env python
  
'''
Welcome to the ArUco Marker Generator!
  
This program:
  - Generates ArUco markers using OpenCV and Python
'''
  
from __future__ import print_function # Python 2/3 compatibility
import cv2 # Import the OpenCV library
import numpy as np # Import Numpy library
  
# Project: ArUco Marker Generator
# Date created: 12/17/2021
# Python version: 3.8
# Reference: https://www.pyimagesearch.com/2020/12/14/generating-aruco-markers-with-opencv-and-python/
 
desired_aruco_dictionary = "DICT_ARUCO_ORIGINAL"
aruco_marker_id = 1
output_filename = "DICT_ARUCO_ORIGINAL_id1.png"
 
# The different ArUco dictionaries built into the OpenCV library. 
ARUCO_DICT = {
  "DICT_4X4_50": cv2.aruco.DICT_4X4_50,
  "DICT_4X4_100": cv2.aruco.DICT_4X4_100,
  "DICT_4X4_250": cv2.aruco.DICT_4X4_250,
  "DICT_4X4_1000": cv2.aruco.DICT_4X4_1000,
  "DICT_5X5_50": cv2.aruco.DICT_5X5_50,
  "DICT_5X5_100": cv2.aruco.DICT_5X5_100,
  "DICT_5X5_250": cv2.aruco.DICT_5X5_250,
  "DICT_5X5_1000": cv2.aruco.DICT_5X5_1000,
  "DICT_6X6_50": cv2.aruco.DICT_6X6_50,
  "DICT_6X6_100": cv2.aruco.DICT_6X6_100,
  "DICT_6X6_250": cv2.aruco.DICT_6X6_250,
  "DICT_6X6_1000": cv2.aruco.DICT_6X6_1000,
  "DICT_7X7_50": cv2.aruco.DICT_7X7_50,
  "DICT_7X7_100": cv2.aruco.DICT_7X7_100,
  "DICT_7X7_250": cv2.aruco.DICT_7X7_250,
  "DICT_7X7_1000": cv2.aruco.DICT_7X7_1000,
  "DICT_ARUCO_ORIGINAL": cv2.aruco.DICT_ARUCO_ORIGINAL
}
  
def main():

  # Check that we have a valid ArUco marker
  if ARUCO_DICT.get(desired_aruco_dictionary, None) is None:
    print("[INFO] ArUCo tag of '{}' is not supported".format(
      args["type"]))
    sys.exit(0)
     
  # Load the ArUco dictionary
  this_aruco_dictionary = cv2.aruco.Dictionary_get(ARUCO_DICT[desired_aruco_dictionary])
   
  # Allocate memory for the ArUco marker
  # We create a 300x300x1 grayscale image, but you can use any dimensions you desire.
  print("[INFO] generating ArUCo tag type '{}' with ID '{}'".format(
    desired_aruco_dictionary, aruco_marker_id))
     
  # Create the ArUco marker
  this_marker = np.zeros((300, 300, 1), dtype="uint8")
  cv2.aruco.drawMarker(this_aruco_dictionary, aruco_marker_id, 300, this_marker, 1)
   
  # Save the ArUco tag to the current directory
  cv2.imwrite(output_filename, this_marker)
  cv2.imshow("ArUco Marker", this_marker)
  cv2.waitKey(0)
   
if __name__ == '__main__':
  print(__doc__)
  main()"""


import cv2
import numpy as np
import base64

# Base64-encoded PNG image of ArUco marker (4x4_50, ID 0)
marker_base64 = """
iVBORw0KGgoAAAANSUhEUgAAASwAAAEuCAIAAACmtwOvAAAAA3NCSVQICAjb4U/gAAAAGXRFWHRT
b2Z0d2FyZQBBZG9iZSBJbWFnZVJlYWR5ccllPAAAAUhJREFUeNrs3cENgDAQRdG1/f8uU3w8Nm8q
kcY8HFMlAEAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAACg
Wg9fD+39r7a0VwPj7ujFujB//9q/d/+/3rbx8xfDPQ3oDfB+g9oDfB+g9oDfB+g9oDfB+g9oDfB+
g9oDfB+g9oDfB+g9oDfB+g9oDfB+g9oDfB+g9oDfB+g9oDfB+g9oDfB+g9oDfB+g9oDfB+g9oDfB
+g9oDfB+g9oDfB+g9oDfB+g9oDfB+g9oDfB+g9oDfB+g9oDfB+g9oDfB+g9oDfB+g9oDfB+g9oDf
B+g9oDfB+g9oDfB+g9oDfB+g9oDfB+g9oDfB+g9oDfB+g9oDfB+g9oDfB+g9oDfB+g9oDfB+g9oD
fB+g9oDfB+g9oDfB+g9oDfB+g9oDfB+g9oDfB+g9oDfB+g9oDfB+g9oDfB+g9oDfB+g9oDfB+g9o
DfB+g9oDfB+g9oDfB+g9oDfB+g9oDfB+g9oDfB+g9oDfB+g9oDfB+g9oDfB+g9oDfB+g9oDfB+g9
oDfB+g9oDfB+g9oDfB+g9oDfB+g9oDfB+g9oDfB+g9oDfB+g9oDfB+g9oDfB+g9oDfB+g9oDfB+g
9oDfB+g9oDfB+g9oDfB+g9oDfB+g9oDfB+g9oDfB+g9oDfB+g9oDfB+g9oDfB+g9oDfB+g9oDfB+
g9oDfB+g9oDfB+g9oDfB+g9oDfB+g9oDfB+g9oDfB+g9oDfB+g9oDfB+g9oDfB+g9oDfB+g9oDfB
+g9oDfB+g9oDfB+g9oDfB+g9oDfB+g9oDfB+g9oDfB+g9oDfB+g9oDfB+g9oDfB+g9oDfB+g9oDf
B+g9oDfB+g9oDfB+g9oDfB+g9oDfB+g9oDfB+g9oDfB+g9oDfB+g9oDfB+g9oDfB+g9oDfB+g9oD
fB+g9oDfB+g9oDfB+g9oDfB+g9oDfB+g9oDfB+g9oDfB+g9oDfB+g9oDfB+g9oDfB+g9oDfB+g9o
DfB+g9oDfB+g9oDfB+g9oDfB+g9oDfB+g9oDfB+g9oDfB+g9oDfB+g9oDfB+g9oDfB+g9oDfB+g9
oDfB+g9oDfB+g9oDfB+g9oDfB+g9oDfB+g9oDfB+g9oDfB+g9oDfB+g9oDfB+g9oDfB+g9oDfB+g
9oDfB+g9oDfB+g9oDfB+g9oDfB+g9oDfB+g9oDfB+g9oDfB+g9oDfB+g9oDfB+g9oDfB+g9oDfB+
g9oDfB+g9oDfB+g9oDfB+g9oDfB+g9oDfB+g9oDfB+g9oDfB+g9oDfB+g9oDfB+g9oDfB+g9oDfB
+g9oDfB+g9oDfB+g9oDfB+g9oDfB+g9oDfB+g9oDfB+g9oDfB+g9oDfB+g9oDfB+g9oDfB+g9oDf
B+g9oDfB+g9oDfB+g9oDfB+g9oDfB+g9oDfB+g9oDfB+g9oDfB+g9oDfB+g9oDfB+g9oDfB+g9oD
fB+g9oDfB+g9oDfB+g9oDfB+g9oDfB+g9oDfB+g9oDfB+g9oDfB+g9oDfB+g9oDfB+g9oDfB+g9o
DfB+g9oDfB+g9oDfB+g9oDfB+g9oDfB+g9oDfB+g9oDfB+g9oDfB+g9oDfB+g9oDfB+g9oDfB+g9
oDfB+g9oDfB+g9oDfB+g9oDfB+g9oDfB+g9oDfB+g9oDfB+g9oDfB+g9oDfB+g9oDfB+g9oDfB+g
9oDfB+g9oDfB+g9oDfB+g9oDfB+g9oDfB+g9oDfB+g9oDfB+g9oDfB+g9oDfB+g9oDfB+g9oDfB+
g9oDfB+g9oDfB+g9oDfB+g9oDfB+g9oDfB+g9oDfB+g9oDfB+g9oDfB+g9oDfB+g9oDfB+g9oDfB
+g9oDfB+g9oDfB+g9oDfB+g9oDfB+g9oDfB+g9oDfB+g9oDfB+g9oDfB+g9oDfB+g9oDfB+g9oDf
B+g9oDfB+g9oDfB+g9oDfB+g9oDfB+g9oDfB+g9oDfB+g9oDfB+g9oDfB+g9oDfB+g9oDfB+g9oD
fB+g9oDfB+g9oDfB+g9oDfB+g9oDfB+g9oDfB+g9oDfB+g9oDfB+g9oDfB+g9oDfB+g9oDfB+g9o
DfB+g9oDfB+g9oDfB+g9oDfB+g9oDfB+g9oDfB+g9oDfB+g9oDfB+g9oDfB+g9oDfB+g9oDfB+g9
oDfB+g9oDfB+g9oDfB+g9oDfB+g9oDfB+g9oDfB+g9oDfB+g9oDfB+g9oDfB+g9oDfB+g9oDfB+g
9oDfB+g9oDfB+g9oDfB+g9oDfB+g9oDfB+g9oDfB+g9oDfB+g9oDfB+g9oDfB+g9oDfB+g9oDfB+
g9oDfB+g9oDfB+g9oDfB+g9oDfB+g9oDfB+g9oDfB+g9oDfB+g9oDfB+g9oDfB+g9oDfB+g9oDfB
+g9oDfB+g9oDfB+g9oDfB+g9oDfB+g9oDfB+g9oDfB+g9oDfB+g9oDfB+g9oDfB+g9oDfB+g9oDf
B+g9oDfB+g9oDfB+g9oDfB+g9oDfB+g9oDfB+g9oDfB+g9oDfB+g9oDfB+g9oDfB+g9oDfB+g9oD
fB+g9oDfB+g9oDfB+g9oDfB+g9oDfB+g9oDfB+g9oDfB+g9oDfB+g9oDfB+g9oDfB+g9oDfB+g9o
DfB+g9oDfB+g9oDfB+g9oDfB+g9oDfB+g9oDfB+g9oDfB+g9oDfB+g9oDfB+g9oDfB+g9oDfB+g9
oDfB+g9oDfB+g9oDfB+g9oDfB+g9oDfB+g9oDfB+g9oDfB+g9oDfB+g9oDfB+g9oDfB+g9oDfB+g
9oDfB+g9oDfB+g9oDfB+g9oDfB+g9oDfB+g9oDfB+g9oDfB+g9oDfB+g9oDfB+g9oDfB+g9oDfB+
g9oDfB+g9oDfB+g9oDfB+g9oDfB+g9oDfB+g9oDfB+g9oDfB+g9oDfB+g9oDfB+g9oDfB+g9oDfB
+g9oDfB+g9oDfB+g9oDfB+g9oDfB+g9oDfB+g9oDfB+g9oDfB+g9oDfB+g9oDfB+g9oDfB+g9oDf
B+g9oDfB+g9oDfB+g9oDfB+g9oDfB+g9oDfB+g9oDfB+g9oDfB+g9oDfB+g9oDfB+g9oDfB+g9oD
fB+g9oDfB+g9oDfB+g9oDfB+g9oDfB+g9oDfB+g9oDfB+g9oDfB+g9oDfB+g9oDfB+g9oDfB+g9o
DfB+g9oDfB+g9oDfB+g9oDfB+g9oDfB+g9oDfB+g9oDfB+g9oDfB+g9oDfB+g9oDfB+g9oDfB+g9
oDfB+g9oDfB+g9oDfB+g9oDfB+g9oDfB+g9oDfB+g9oDfB+g9oDfB+g9oDfB+g9oDfB+g9oDfB+g
9oDfB+g9oDfB+g9oDfB+g9oDfB+g9oDfB+g9oDfB+g9oDfB+g9oDfB+g9oDfB+g9oDfB+g9oDfB+
g9oDfB+g9oDfB+g9oDfB+g9oDfB+g9oDfB+g9oDfB+g9oDfB+g9oDfB+g9oDfB+g9oDfB+g9oDfB
+g9oDfB+g9oDfB+g9oDfB+g9oDfB+g9oDfB+g9oDfB+g9oDfB+g9oDfB+g9oDfB+g9oDfB+g9oDf
B+g9oDfB+g9oDfB+g9oDfB+g9oDfB+g9oDfB+g9oDfB+g9oDfB+g9oDfB+g9oDfB+g9oDfB+g9oD
fB+g9oDfB+g9oDfB+g9oDfB+g9oDfB+g9oDfB+g9oDfB+g9oDfB+g9oDfB+g9oDfB+g9oDfB+g9o
DfB+g9oDfB+g9oDfB+g9oDfB+g9oDfB+g9oDfB+g9oDfB+g9oDfB+g9oDfB+g9oDfB+g9oDfB+g9
oDfB+g9oDfB+g9oDfB+g9oDfB+g9oDfB+g9oDfB+g9oDfB+g9oDfB+g9oDfB+g9oDfB+g9oDfB+g
9oDfB+g9oDfB+g9oDfB+g9oDfB+g9oDfB+g9oDfB+g9oDfB+g9oDfB+g9oDfB+g9oDfB+g9oDfB+
g9oDfB+g9oDfB+g9oDfB+g9oDfB+g9oDfB+g9oDfB+g9oDfB+g9oDfB+g9oDfB+g9oDfB+g9oDfB
+g9oDfB+g9oDfB+g9oDfB+g9oDfB+g9oDfB+g9oDfB+g9oDfB+g9oDfB+g9oDfB+g9oDfB+g9oDf
B+g9oDfB+g9oDfB+g9oDfB+g9oDfB+g9oDfB+g9oDfB+g9oDfB+g9oDfB+g9oDfB+g9oDfB+g9oD
fB+g9oDfB+g9oDfB+g9oDfB+g9oDfB+g9oDfB+g9oDfB+g9oDfB+g9oDfB+g9oDfB+g9oDfB+g9o
"""

# Decode base64 to image
# Assume `marker_base64` is your Base64 string
# Add padding if necessary
missing_padding = len(marker_base64) % 4
if missing_padding != 0:
    marker_base64 += '=' * (4 - missing_padding)

img_data = base64.b64decode(marker_base64)

img_data = base64.b64decode(marker_base64)
nparr = np.frombuffer(img_data, np.uint8)
image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# ArUco detection
dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
parameters = cv2.aruco.DetectorParameters()
detector = cv2.aruco.ArucoDetector(dictionary, parameters)
corners, ids, _ = detector.detectMarkers(gray)

if ids is not None:
    print(f"Detected marker IDs: {ids.flatten()}")
    cv2.aruco.drawDetectedMarkers(image, corners, ids)

    for marker_corners in corners:
        pts = marker_corners[0]
        cx = int(np.mean(pts[:, 0]))
        cy = int(np.mean(pts[:, 1]))
        print(f"Centroid: ({cx}, {cy})")
        cv2.circle(image, (cx, cy), 5, (0, 255, 0), -1)
else:
    print("No markers detected.")

cv2.imshow("Marker Detection", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
