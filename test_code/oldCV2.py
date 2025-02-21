import cv2
import numpy as np
import os

# cv2.getBuildInformation()
print(cv2.getBuildInformation())

def remove_black_bar(image):
    """Automatically detects and removes the bottom black bar in an image."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
    
    # Find contours to locate the black bar
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if h > image.shape[0] * 0.05:  # If height of detected bar is significant
            image = image[:y, :]
            break
    return image

def segment_features(image):
    """Basic segmentation using color thresholding to highlight green leaves."""
    # Convert to HSV using CUDA
    gpu_image = cv2.cuda_GpuMat()
    gpu_image.upload(image)

    gpu_hsv = cv2.cuda.cvtColor(gpu_image, cv2.COLOR_BGR2HSV)

    # Convert back to CPU for thresholding
    hsv = gpu_hsv.download()
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower_green = np.array([35, 40, 40])
    upper_green = np.array([85, 255, 255])
    mask = cv2.inRange(hsv, lower_green, upper_green)
    result = cv2.bitwise_and(image, image, mask=mask)
    return result

def preprocess_image(image_path, output_dir):
    """Loads an image, removes black bar, segments features, and saves the result."""
    image = cv2.imread(image_path)
    image = remove_black_bar(image)
    # Check if image is loaded properly
    if image is None:
        print(f"Error: Could not load image {image_path}")
        return
    segmented = segment_features(image)
    
    filename = os.path.basename(image_path)
    cv2.imwrite(os.path.join(output_dir, f"processed_{filename}"), segmented)

def process_folder(input_folder, output_folder):
    """Processes all images in a folder."""
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    for filename in os.listdir(input_folder):
        if filename.lower().endswith(('png', 'jpg', 'jpeg')):
            preprocess_image(os.path.join(input_folder, filename), output_folder)

# Example Usage
input_folder = "/media/downina3/F6F2-62B6/SmallLance"
output_folder = "/media/downina3/F6F2-62B6/SmallLance_PrePros"
process_folder(input_folder, output_folder)

