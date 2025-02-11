import numpy as np
import cv2
import torch
import time

# Load YOLOv5 Model
model = torch.hub.load('ultralytics/yolov5', 'custom', path='best.pt', device='0')

# Video source
VIDEO_URL = '/dev/video2'

# Initialize global variables
global sx, sy
sx, sy = 210, 210

# Delay before starting
time.sleep(5)

# Open video capture
cap = cv2.VideoCapture(VIDEO_URL)

# Define bounding box dimensions
x, y = (640, 480)
l, b = (140, 140)
mid_x, mid_y = (210, 210)

p1 = (int(mid_x - l), int(mid_y - b))
p2 = (int(mid_x + l), int(mid_y + b))

# Mouse click event callback
def onMouse(event, x, y, flags, param):
    global sx, sy
    if event == cv2.EVENT_LBUTTONDOWN:
        sx, sy = x, y
        print(f'x = {x}, y = {y}')

# Setup OpenCV window
cv2.namedWindow('frame')
cv2.setMouseCallback('frame', onMouse)

while True:
    ret, frameI = cap.read()
    if not ret:
        break

    # Crop frame
    frame = frameI[24:452, 98:520]
    im = frame[..., ::-1]  # Convert BGR to RGB
    
    # Run inference
    results = model(im, size=360)
    im0 = cv2.cvtColor(np.array(results.render()[0]), cv2.COLOR_BGR2RGB)
    predictions = results.pandas().xyxy[0]

    for _, row in predictions.iterrows():
        xmin, xmax = row["xmin"], row["xmax"]
        ymin, ymax = row["ymin"], row["ymax"]

        xd = int(xmin + ((xmax - xmin) / 2))
        yd = int(ymin + ((ymax - ymin) / 2))

        print(f"Center - ({xd}, {sx})")
        print("--------------------------------------------")

    # Draw bounding box and center points
    cv2.rectangle(im0, p1, p2, (255, 0, 0), 2)
    cv2.circle(im0, (mid_x, mid_y), 3, (0, 0, 255), 4)
    cv2.circle(im0, (sx, sy), 3, (0, 255, 0), 4)

    cv2.imshow('frame', im0)

    # Exit on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
