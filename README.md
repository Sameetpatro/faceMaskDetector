# Face Mask Detection using Deep Learning

This project can **detect whether a person is wearing a face mask or not** using a webcam or image.

It uses **Deep Learning (MobileNetV2)** and **OpenCV** to work in real time.

---

## What this project does

The program:

1. Looks at a person’s face through a webcam or an image.
2. Detects if the person’s face has a **mask** or **no mask**.
3. Shows the result on the screen with a label and box around the face.

---

## How it works

1. **Dataset** — Around 4000 images of people with and without masks were collected.

2. **Model Training** —

   * The images are split into 80% training and 20% testing.
   * A **MobileNetV2** deep learning model (pre-trained on ImageNet) is used.
   * Only the last few layers are trained to detect masks.

3. **Saved Model** — After training, a file called `mask_detector_model.h5` is created.

4. **Detection** —

   * The webcam or an image is given to the model.
   * It checks the face area and predicts “Mask” or “No Mask”.

---

## 🖥️ How to run the project

### Step 1: Install the requirements

Make sure you have Python installed, then open Command Prompt in your project folder and type:

```bash
pip install -r requirements.txt
```

### Step 2: Train your own model (optional)

If you want to train it yourself:

```bash
python train_model.py
```

This will create a new model file — `mask_detector_model.h5`.

### Step 3: Run the detector

After the model is ready:

```bash
python mask_detection_main.py
```

It will open your webcam and start detecting faces with or without masks.

---

---

## 🧪 Technologies Used

* **Python 3**
* **TensorFlow / Keras**
* **OpenCV**
* **MobileNetV2 (Pretrained CNN)**
* **NumPy & Matplotlib**

---

## ⚡ Features

✅ Works in real time using a webcam
✅ Detects faces accurately
✅ Trained on a custom dataset
✅ Can easily be improved with more mask types and colors

---

## 💡 Future Improvements

* Add third category → “Improperly worn mask”
* Improve accuracy with more training images
* Deploy the model as a web or mobile app

---

## 👨‍💻 Made By

**Sameet Patro**
Undergraduate Student — IIIT Sonepat

