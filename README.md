# Face Mask Detection using OpenCV and CNN

A simple and effective deep learning project that detects whether a person is wearing a mask or not in real time using a webcam feed.

---

## 🔍 Overview

This project uses a pre-trained CNN model (`mask_detector_model.h5`) combined with OpenCV’s Haar Cascade classifier (`face_detector.xml`) to detect faces and classify them as **Mask** or **No Mask**.

---

## ⚙️ How It Works

1. **Face Detection:**

   * The Haar Cascade classifier locates faces in the video frame.
2. **Feature Extraction & Classification:**

   * Each detected face is resized and passed into the CNN model.
   * The model predicts whether the face has a mask or not.
3. **Real-Time Feedback:**

   * Bounding boxes (green for mask, red for no mask) are drawn on the live video feed.

---

## 🧠 Technologies Used

* Python 3.x
* OpenCV
* TensorFlow / Keras
* NumPy

---

## ▶️ How to Run

1. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```
2. Run the main script:

   ```bash
   python mask_detection_main.py
   ```
3. The webcam feed will open and start detecting faces with or without masks.

---

## 📁 Folder Structure

```
FACEMASKDETECTOR/
│
├── dataset_train/         # Training images
├── dataset_test/          # Testing images
├── mask_detector_model.h5 # Pretrained CNN model
├── face_detector.xml      # Haar Cascade for face detection
├── mask_detection_main.py # Main detection script
├── requirements.txt       # Required libraries
└── sample_images/         # Example images (mask.png, nomask.png)
```

---

## 🚀 Possible Improvements

* Add a third class: *Improperly Worn Mask*
* Train on a larger dataset with pose variation and occlusion
* Integrate Flask or Streamlit for a web dashboard

---
