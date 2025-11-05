import os, shutil, random
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


base_dir = "dataset"         
train_dir = "dataset_train"
test_dir = "dataset_test"


def prepare_split():
    if os.path.exists(train_dir): return
    os.makedirs(train_dir + "/mask", exist_ok=True)
    os.makedirs(train_dir + "/nomask", exist_ok=True)
    os.makedirs(test_dir + "/mask", exist_ok=True)
    os.makedirs(test_dir + "/nomask", exist_ok=True)

    for cls in ["mask", "nomask"]:
        images = os.listdir(os.path.join(base_dir, cls))
        train_imgs, test_imgs = train_test_split(images, test_size=0.2, random_state=42)
        for img in train_imgs:
            shutil.copy(os.path.join(base_dir, cls, img), os.path.join(train_dir, cls, img))
        for img in test_imgs:
            shutil.copy(os.path.join(base_dir, cls, img), os.path.join(test_dir, cls, img))
    print("Dataset split into train/test folders.")

prepare_split()

train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=25,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.15,
    zoom_range=0.25,
    brightness_range=[0.5, 1.5],
    horizontal_flip=True,
    fill_mode='nearest'
)
test_datagen = ImageDataGenerator(rescale=1./255)

train_gen = train_datagen.flow_from_directory(train_dir, target_size=(150,150),
                                              batch_size=32, class_mode='binary')
test_gen = test_datagen.flow_from_directory(test_dir, target_size=(150,150),
                                            batch_size=32, class_mode='binary')


base = MobileNetV2(weights='imagenet', include_top=False, input_shape=(150,150,3))
base.trainable = False

model = Sequential([
    base,
    GlobalAveragePooling2D(),
    Dense(128, activation='relu'),
    Dropout(0.4),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer=Adam(1e-4), loss='binary_crossentropy', metrics=['accuracy'])
print(model.summary())


history = model.fit(train_gen, validation_data=test_gen, epochs=8)
model.save("mask_detector_model.h5")
print("Model trained and saved.")


plt.plot(history.history['accuracy'], label='Train')
plt.plot(history.history['val_accuracy'], label='Val')
plt.legend(); plt.xlabel('Epoch'); plt.ylabel('Accuracy'); plt.title('Training Progress')
plt.show()
