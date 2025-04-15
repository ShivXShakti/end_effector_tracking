import cv2
import numpy as np
import cairosvg
from io import BytesIO

svg_path = '/home/scg/Documents/kd_ws/urs_ws/src/end_effector_tracking/data/DICT_4X4_50.svg'
with open(svg_path, 'rb') as f:
    png_data = BytesIO()
    #cairosvg.svg2png(file_obj=f, write_to=png_data)
    cairosvg.svg2png(file_obj=f, write_to=png_data, output_width=1000, output_height=1000)

    #cairosvg.svg2png(file_obj=f, write_to=png_data, output_width=600, output_height=600)
    nparr = np.frombuffer(png_data.getvalue(), np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
#image = cv2.imread("debug_marker.png")  # A real ArUco image file
# Check if aruco is part of cv2
if not hasattr(cv2, 'aruco'):
    raise ImportError("OpenCV-contrib-python is not installed properly with ArUco support.")

# Load image
#image = cv2.imread('data/4x4_50_0.png')  # Make sure this is a PNG/JPG image
if image is None:
    raise ValueError("Image not found. Check the path and file format.")

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
cv2.imwrite("debug_color.png", image)
cv2.imwrite("debug_gray.png", gray)


# For OpenCV >= 4.7 use the new ArucoDetector class
dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
#parameters = cv2.aruco.DetectorParameters()
parameters = cv2.aruco.DetectorParameters()
parameters.adaptiveThreshWinSizeMin = 3
parameters.adaptiveThreshWinSizeMax = 23
parameters.adaptiveThreshWinSizeStep = 10
parameters.minMarkerPerimeterRate = 0.01
parameters.maxMarkerPerimeterRate = 4.0
parameters.polygonalApproxAccuracyRate = 0.05
parameters.minCornerDistanceRate = 0.05
parameters.minDistanceToBorder = 2
parameters.minMarkerDistanceRate = 0.01
parameters.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX


detector = cv2.aruco.ArucoDetector(dictionary, parameters)
corners, ids, _ = detector.detectMarkers(gray)
print(f"dictionary: {dictionary},   parameters: {parameters},  detector: {detector}, ids: {ids}")


if ids is not None:
    print(f"Detected marker IDs: {ids.flatten()}")
    cv2.aruco.drawDetectedMarkers(image, corners, ids)

    for marker_corners in corners:
        pts = marker_corners[0]  # shape (4,2)
        cx = int(np.mean(pts[:, 0]))
        cy = int(np.mean(pts[:, 1]))
        print(f"cx: {cx}, cy:{cy}")

        # Draw centroid
        cv2.circle(image, (cx, cy), 5, (0, 255, 0), -1)
        cv2.putText(image, f'Centroid: ({cx}, {cy})', (cx + 10, cy),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 1, cv2.LINE_AA)
else:
    print("No markers detected.")

# Display
cv2.imshow("Detected ArUco Marker", gray)
print("Press any key in the image window to exit...")
while True:
    if cv2.waitKey(10) != -1:  # Wait for any key press
        break
cv2.destroyAllWindows()

