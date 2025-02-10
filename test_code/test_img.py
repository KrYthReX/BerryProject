import cv2
import numpy as np

# Load image
image = cv2.imread('/home/downina3/Downloads/WSCT1045.JPG')

# Convert to HSV for better color handling
hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
l, a, b = cv2.split(lab)
clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
l_clahe = clahe.apply(l)
lab_clahe = cv2.merge((l_clahe, a, b))
enhanced_image = cv2.cvtColor(lab_clahe, cv2.COLOR_LAB2BGR)

# # Save or display the enhanced image
# cv2.imshow('Enhanced Image', enhanced_image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()



# Define desired display dimensions
display_width = 1200  # Width of the resized window
display_height = 1000  # Height of the resized window

# Resize the enhanced image for display
enhanced_image_resized = cv2.resize(enhanced_image, (display_width, display_height))

# Display the resized image
cv2.imshow('Enhanced Image', enhanced_image_resized)
cv2.waitKey(0)
cv2.destroyAllWindows()



# Define pink color range in HSV
lower_pink = np.array([200, 30, 100])  # Adjust for hue, saturation, value
upper_pink = np.array([170, 255, 255])

# Mask for pink ribbon
mask = cv2.inRange(hsv_image, lower_pink, upper_pink)
result = cv2.bitwise_and(image, image, mask=mask)

# Find contours of the ribbon
contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

if contours:
    # Get the largest contour
    ribbon_contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(ribbon_contour)
    print(y)
    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)  # Draw bounding box
# 
# Save or display the result
enhanced_image_resized = cv2.resize(enhanced_image, (display_width, display_height))
cv2.imshow('Ribbon Detection', enhanced_image_resized)
cv2.waitKey(0)
cv2.destroyAllWindows()



# Define region of interest (ROI) around the ribbon
roi = image[y:y+h, x:x+w]

# Optional: Expand ROI size to include nearby branches
padding = 50  # Add padding around the ribbon
roi_expanded = image[max(0, y-padding):y+h+padding, max(0, x-padding):x+w+padding]

# Display the ROI
cv2.imshow('Region of Interest', roi_expanded)
cv2.waitKey(0)
cv2.destroyAllWindows()


# untested portion here
# #----------------------------------------------------------------
# from ultralytics import YOLO

# # Load pretrained YOLO model (or train your own on bud images)
# model = YOLO("yolov8n.pt")  # Replace with a custom model trained on buds

# # Run detection on the ROI
# results = model.predict(roi_expanded)

# # Display results with bounding boxes
# results[0].plot()

