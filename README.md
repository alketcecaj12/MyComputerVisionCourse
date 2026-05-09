# 👁️ My Computer Vision Course

> *Teaching machines to see — one pixel at a time.*

![Python](https://img.shields.io/badge/Python-3.8%2B-3776AB?style=flat-square&logo=python&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-4.x-5C3EE8?style=flat-square&logo=opencv&logoColor=white)
![YOLO](https://img.shields.io/badge/YOLO-Ultralytics-FF6600?style=flat-square)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626?style=flat-square&logo=jupyter&logoColor=white)
![Weeks](https://img.shields.io/badge/Duration-10%20Weeks-22c55e?style=flat-square)
![Level](https://img.shields.io/badge/Level-Beginner%20→%20Intermediate-6366f1?style=flat-square)

---

## 🎯 About This Course

This is a **hands-on, 10-week Python course** in computer vision — built around real code, real images, and real-time inference. It covers the full arc from raw pixel manipulation to deploying a live YOLO object detection pipeline, entirely within Jupyter notebooks.

No slides. No theory dumps. Just progressively deeper concepts, one notebook per week, each building on the last.

The course uses **OpenCV** as the backbone for classical computer vision and **Ultralytics YOLO** for state-of-the-art deep learning object detection — the same stack used in production systems today.

---

## 🗺️ Learning Arc

```
Weeks 1–3 │ Foundation      → What is an image? How does Python see it?
Weeks 4–5 │ Transformation  → Resize, rotate, filter, smooth, detect edges
Weeks 7–8 │ Detection       → Find shapes, faces, and objects (classical + YOLO)
Weeks 9–10│ Real-time       → Video streams, live inference, full pipelines
```

---

## 🗓️ Week-by-Week Curriculum

---

### 📘 Week 1 — Python & Image Fundamentals

**The core idea:** An image is just a matrix of numbers. Once you internalize that, everything else follows.

In OpenCV, a colour image is a NumPy array of shape `(height, width, 3)` — three channels (Blue, Green, Red in OpenCV's channel order). A grayscale image is `(height, width)`. Every pixel is just an integer between 0 and 255.

**Key concepts covered:**
- Setting up the OpenCV + NumPy + Matplotlib environment
- Loading an image with `cv2.imread()` and displaying it with `cv2.imshow()` or `plt.imshow()`
- Understanding image shape, dtype, and memory layout
- The BGR vs RGB distinction — why OpenCV uses BGR and when it matters
- Converting between colour spaces: `cv2.cvtColor(img, cv2.COLOR_BGR2RGB)`

**Code pattern you'll master:**
```python
import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread('image.jpg')
print(img.shape)   # (480, 640, 3)
print(img.dtype)   # uint8

img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
plt.imshow(img_rgb)
plt.axis('off')
plt.show()
```

---

### 📘 Week 2 — Image I/O & Properties

**The core idea:** Before you can process images, you need to read, write, and understand them precisely.

**Key concepts covered:**
- Reading images: `cv2.imread()` with flags (`cv2.IMREAD_GRAYSCALE`, `cv2.IMREAD_COLOR`, `cv2.IMREAD_UNCHANGED`)
- Writing images: `cv2.imwrite()` and format considerations (JPEG lossy vs PNG lossless)
- Image metadata: width, height, channels, bit depth
- Splitting and merging colour channels: `cv2.split()` and `cv2.merge()`
- Exploring pixel value distributions with histograms: `cv2.calcHist()`

**Code pattern you'll master:**
```python
# Split BGR channels and inspect each
b, g, r = cv2.split(img)

# Plot histogram for the red channel
plt.hist(r.ravel(), bins=256, range=[0, 256], color='red', alpha=0.6)
plt.xlabel('Pixel intensity')
plt.ylabel('Count')
plt.title('Red channel histogram')
plt.show()
```

---

### 📘 Week 3 — Pixels, ROI & Creating Images

**The core idea:** You can read, write, and arithmetically combine individual pixels — and carve out any rectangular sub-region of an image.

**Key concepts covered:**
- Direct pixel access: `img[y, x]` for colour, `img[y, x, channel]` for single channel
- Region of Interest (ROI): `roi = img[y1:y2, x1:x2]`
- Copying and pasting ROIs into other images
- Creating synthetic images from scratch with `np.zeros()` and `np.ones()`
- Drawing on images: `cv2.rectangle()`, `cv2.circle()`, `cv2.line()`, `cv2.putText()`

**Code pattern you'll master:**
```python
# Extract a region of interest
roi = img[100:300, 200:400]

# Create a blank canvas and paste the ROI into it
canvas = np.zeros_like(img)
canvas[50:250, 50:250] = roi

# Draw a rectangle on the original image
cv2.rectangle(img, (200, 100), (400, 300), color=(0, 255, 0), thickness=2)
```

---

### 📘 Week 4 — Geometric Transformations

**The core idea:** Every spatial manipulation of an image — resize, rotate, flip, warp — is a mathematical operation on the pixel coordinate grid.

**Key concepts covered:**
- Resizing: `cv2.resize()` with interpolation methods (`INTER_LINEAR`, `INTER_CUBIC`, `INTER_NEAREST`)
- Flipping: `cv2.flip()` horizontally, vertically, and both axes
- Rotation: `cv2.getRotationMatrix2D()` + `cv2.warpAffine()`
- Affine transformations: mapping 3 point pairs to shear, translate, scale simultaneously
- Perspective transforms: `cv2.getPerspectiveTransform()` + `cv2.warpPerspective()` — the technique behind document scanners

**Code pattern you'll master:**
```python
# Rotate image 45 degrees around its centre
h, w = img.shape[:2]
centre = (w // 2, h // 2)
M = cv2.getRotationMatrix2D(centre, angle=45, scale=1.0)
rotated = cv2.warpAffine(img, M, (w, h))

# Perspective warp (e.g. straighten a tilted document)
src_pts = np.float32([[x1,y1],[x2,y2],[x3,y3],[x4,y4]])
dst_pts = np.float32([[0,0],[w,0],[0,h],[w,h]])
M_persp = cv2.getPerspectiveTransform(src_pts, dst_pts)
warped = cv2.warpPerspective(img, M_persp, (w, h))
```

---

### 📘 Week 5 — Filtering, Smoothing & Edge Detection

**The core idea:** Filters are the workhorses of image processing — they suppress noise, sharpen details, and reveal structure. Edge detection is where images start to "mean" something.

**Key concepts covered:**

*Smoothing / blurring:*
- Averaging blur: `cv2.blur()` — simple and fast
- Gaussian blur: `cv2.GaussianBlur()` — weighted, preserves edges better
- Median blur: `cv2.medianBlur()` — excellent for salt-and-pepper noise
- Bilateral filter: `cv2.bilateralFilter()` — edge-preserving smoothing

*Edge detection:*
- Gradient-based: Sobel (`cv2.Sobel()`) and Laplacian (`cv2.Laplacian()`) operators
- Canny edge detector: `cv2.Canny()` — the gold standard, with hysteresis thresholding
- Morphological operations: erosion, dilation, opening, closing via `cv2.morphologyEx()`

**Code pattern you'll master:**
```python
# Canny edge detection pipeline
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(gray, (5, 5), 0)
edges = cv2.Canny(blurred, threshold1=50, threshold2=150)

# Morphological closing to fill gaps in edges
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
```

---

### 📘 Week 7 — Contours & Shape Analysis

**The core idea:** Contours are the boundaries of objects in a binary image — they're what bridges edge detection to understanding *what* and *where* things are.

**Key concepts covered:**
- Finding contours: `cv2.findContours()` with retrieval modes and approximation methods
- Drawing contours: `cv2.drawContours()`
- Contour properties: area (`cv2.contourArea()`), perimeter (`cv2.arcLength()`), bounding box (`cv2.boundingRect()`), centroid via moments
- Shape approximation: `cv2.approxPolyDP()` — simplify contours to polygons
- Convex hull: `cv2.convexHull()` — tightest convex envelope of a shape
- Filtering contours by area to remove noise

**Code pattern you'll master:**
```python
# Full contour extraction pipeline
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
_, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

for cnt in contours:
    area = cv2.contourArea(cnt)
    if area > 500:  # filter small noise
        x, y, w, h = cv2.boundingRect(cnt)
        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)

        M = cv2.moments(cnt)
        if M['m00'] != 0:
            cx = int(M['m10'] / M['m00'])
            cy = int(M['m01'] / M['m00'])
            cv2.circle(img, (cx, cy), 5, (255, 0, 0), -1)
```

---

### 📘 Week 8 — Object Detection: Haar Cascades & YOLO

**The core idea:** Detection = finding *where* an object is. Week 8 covers both the classical approach (Haar cascades) and the modern deep learning approach (YOLO), so you understand the full evolution.

**Haar Cascades (classical):**
- Pre-trained XML classifiers for faces, eyes, cars, etc.
- Sliding window + scale pyramid approach
- `cv2.CascadeClassifier()` — fast but limited to trained categories

**YOLO (deep learning):**
- You Only Look Once — single forward pass, predicts bounding boxes + class labels simultaneously
- Ultralytics YOLOv8/YOLOv11: modern, pip-installable, pretrained on COCO (80 classes)
- Confidence scores and Non-Maximum Suppression (NMS) to remove duplicate boxes

**Code pattern you'll master:**
```python
# Classical: Haar face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
for (x, y, w, h) in faces:
    cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)

# Modern: YOLO object detection
from ultralytics import YOLO
model = YOLO('yolov8n.pt')   # nano model — fast and lightweight
results = model.predict(source='image.jpg', conf=0.5)
results[0].show()
```

**COCO classes detected out-of-the-box:** person, car, truck, bus, bicycle, motorcycle, dog, cat, chair, laptop, phone, and 70 more.

---

### 📘 Week 9 — Video Processing & Real-Time Inference

**The core idea:** Video is just a sequence of images. Everything you've learned about images applies frame-by-frame — the challenge is doing it fast enough.

**Key concepts covered:**
- Opening video files and webcam streams: `cv2.VideoCapture()`
- Reading frames in a loop: `cap.read()` returns `(ret, frame)`
- Writing processed video: `cv2.VideoWriter()`
- Frame-by-frame YOLO inference on a video stream
- FPS measurement and performance profiling
- Graceful exit with keypress detection: `cv2.waitKey(1) & 0xFF == ord('q')`

**Code pattern you'll master:**
```python
from ultralytics import YOLO
import cv2

model = YOLO('yolov8n.pt')
cap = cv2.VideoCapture(0)  # 0 = webcam, or pass a video file path

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    results = model.predict(source=frame, conf=0.4, verbose=False)
    annotated = results[0].plot()  # draws boxes + labels on the frame

    cv2.imshow('Live Detection', annotated)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
```

---

### 📘 Week 10 — Capstone & Advanced Topics

**The core idea:** Consolidation week — combining the full pipeline, plus a look at what comes next.

**Key concepts covered:**
- End-to-end pipeline: image → preprocessing → detection → annotation → output
- Object tracking across frames (intro to `cv2.TrackerCSRT_create()`)
- Custom inference logic: filtering detections by class, size, or position
- Saving annotated outputs — both images and video
- Where to go next: fine-tuning YOLO on custom datasets, semantic segmentation, pose estimation

---

## 🧰 Tech Stack

| Library | Version | Purpose |
|---------|---------|---------|
| `opencv-python` | 4.x | Core image processing, I/O, drawing, classical CV |
| `numpy` | 1.24+ | Array math, pixel manipulation, matrix operations |
| `ultralytics` | 8.x | YOLOv8/v11 object detection, training, inference |
| `matplotlib` | 3.x | Visualization, histogram plotting, figure display |
| `jupyter` | latest | Interactive notebook environment |

---

## 🚀 Quick Start

### 1. Clone the repository

```bash
git clone https://github.com/alketcecaj12/MyComputerVisionCourse.git
cd MyComputerVisionCourse
```

### 2. Create a virtual environment (recommended)

```bash
python -m venv cv-env
source cv-env/bin/activate       # macOS / Linux
cv-env\Scripts\activate          # Windows
```

### 3. Install dependencies

```bash
pip install opencv-python numpy matplotlib jupyter ultralytics
```

### 4. Launch the first notebook

```bash
cd week1_python
jupyter notebook
```

Work through the weeks in order — each one builds directly on the previous.

---

## 📋 Prerequisites

| Requirement | Notes |
|-------------|-------|
| Python 3.8+ | Anaconda distribution recommended |
| Basic Python | Functions, loops, lists, dictionaries |
| NumPy basics | Helpful but not required — covered as needed |
| Prior CV experience | **Not required** — zero assumed |
| GPU | Not required — all weeks run on CPU |

---

## 📁 Repository Structure

```
MyComputerVisionCourse/
│
├── week1_python/     # Image basics — arrays, shape, colour spaces
├── week2_python/     # Image I/O — read, write, channels, histograms
├── week3-python/     # Pixels and ROI — access, draw, create
├── week4-python/     # Geometric transforms — resize, rotate, warp
├── week5_python/     # Filtering & edge detection — blur, Canny, morphology
├── week7-python/     # Contours & shape analysis
├── week8-python/     # Object detection — Haar cascades + YOLO
├── week9-python/     # Video & real-time inference
└── week10-python/    # Capstone — end-to-end pipeline
```

> **Note:** Week 6 is not included — the course jumps from Week 5 to Week 7, which covers contour-based shape analysis.

---

## 🏁 What You'll Be Able to Build by the End

By completing this course you will be able to:

- ✅ Load, manipulate, display, and save images in any format
- ✅ Apply filters and detect edges with Gaussian, Median, and Canny operators
- ✅ Perform geometric transforms — resize, rotate, affine warp, perspective correction
- ✅ Extract contours, compute shape properties, and classify simple shapes
- ✅ Detect objects in images using both Haar cascades and YOLO
- ✅ Process video streams frame-by-frame with real-time YOLO inference
- ✅ Build a complete end-to-end computer vision pipeline from raw input to annotated output

---

## ⚡ Difficulty Progression

```
Week 1  ████░░░░░░  Beginner      — Images as arrays
Week 2  ████░░░░░░  Beginner      — I/O and metadata
Week 3  █████░░░░░  Beginner+     — Pixels and drawing
Week 4  ██████░░░░  Intermediate  — Math of transformations
Week 5  ███████░░░  Intermediate  — Convolutions and edge theory
Week 7  ███████░░░  Intermediate  — Contour geometry
Week 8  ████████░░  Intermediate+ — Classical + deep learning detection
Week 9  █████████░  Advanced      — Real-time pipeline engineering
Week 10 ██████████  Advanced      — Full pipeline + next steps
```

---

## 🛠️ Troubleshooting

**`cv2.imshow()` doesn't work in Jupyter**
Use `plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))` instead. OpenCV's window functions don't work reliably in notebook environments.

**YOLO model download fails**
The first call to `YOLO('yolov8n.pt')` downloads the weights (~6MB). Ensure you have internet access. Weights are cached in `~/.ultralytics/`.

**Webcam not opening (Week 9)**
Try `cv2.VideoCapture(1)` instead of `0` if you have multiple cameras. On some systems you may need `cv2.VideoCapture(0, cv2.CAP_DSHOW)` on Windows.

**Images appear with wrong colours**
Remember: OpenCV loads as BGR, Matplotlib expects RGB. Always convert: `cv2.cvtColor(img, cv2.COLOR_BGR2RGB)` before `plt.imshow()`.

---

## 🔗 Going Further

After completing this course, natural next steps are:

- **Custom YOLO training** — fine-tune on your own labelled dataset with Ultralytics
- **Semantic segmentation** — pixel-level classification with SAM or YOLOv8-seg
- **Pose estimation** — human skeleton keypoints with YOLOv8-pose
- **Deep learning foundations** — CNNs with PyTorch or TensorFlow
- **Deployment** — wrapping your CV pipeline in a Streamlit or FastAPI app

---

## 📬 Author

**Alket Cecaj** — Data Scientist & Quantitative Analyst  
[GitHub](https://github.com/alketcecaj12) · Copenhagen, Denmark

---

*Built with 🐍 Python · 📷 OpenCV · ⚡ YOLO · 📓 Jupyter*
