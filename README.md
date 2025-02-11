# YOLOv5 Object Detection with OpenCV

## Overview
This project utilizes **YOLOv5** for object detection using a connected camera. The system captures video frames, processes them with YOLOv5, and displays the detected objects with bounding boxes. Users can also interact with the system by clicking on the video feed to select a point of interest.

## Features
- Uses **YOLOv5** for real-time object detection.
- Reads video frames from a camera (`/dev/video2`).
- Displays detected objects with bounding boxes.
- Allows user interaction via mouse clicks to mark points of interest.
- Exits gracefully when the user presses `q`.

## Requirements
- Python 3.x
- OpenCV (`cv2`)
- PyTorch
- YOLOv5 model (`best.pt`)
- NumPy

## Installation
1. Clone this repository:
   ```sh
   git clone https://github.com/your-repo/yolov5-object-detection.git
   cd yolov5-object-detection
   ```
2. Install dependencies:
   ```sh
   pip install numpy opencv-python torch torchvision torchaudio pandas
   ```
3. Download the YOLOv5 model (`best.pt`) and place it in the project folder.

## Usage
1. Run the script:
   ```sh
   python detect.py
   ```
2. The program will:
   - Open the camera feed.
   - Run YOLOv5 on each frame to detect objects.
   - Display the processed frame with bounding boxes and a central reference point.
   - Allow users to click on the frame to mark a point.
3. Press `q` to exit the program.

## Code Breakdown
- **Model Loading:** YOLOv5 is loaded using `torch.hub.load()`.
- **Video Capture:** OpenCV reads frames from the specified camera.
- **Inference:** The model detects objects in each frame.
- **Display:** Bounding boxes and user-selected points are shown on the video.
- **Exit Handling:** The program stops when the user presses `q`.

## License
This project is licensed under the MIT License.

## Acknowledgments
- [Ultralytics YOLOv5](https://github.com/ultralytics/yolov5) for the object detection model.
- OpenCV and PyTorch for image processing and deep learning support.

---
Feel free to contribute or modify the script as needed!

