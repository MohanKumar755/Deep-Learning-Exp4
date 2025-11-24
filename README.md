# Deep-Learning-Exp4

## **Implement a Transfer Learning concept in Image Classification**

## **AIM**

To develop an image classification model using transfer learning with the MobileNetV2 architecture for the CIFAR-10 dataset.

## **THEORY**

Transfer Learning is a deep learning approach that reuses a pre-trained model trained on a large dataset (like ImageNet) to solve a new problem with a smaller dataset. Instead of training a neural network from scratch, we “transfer” the learned weights and fine-tune them for the new classification task.

MobileNetV2 is a lightweight CNN designed for mobile and embedded vision applications. It uses depthwise separable convolutions and inverted residual blocks, which significantly reduce the number of parameters while maintaining high accuracy.

**Advantages of Transfer Learning:**

- Speeds up training by leveraging pre-trained weights.

- Reduces the risk of overfitting on smaller datasets.

- Achieves higher accuracy with fewer data and resources.

## **DESIGN STEPS**

**STEP 1:**

Import TensorFlow and Keras libraries required for transfer learning.

**STEP 2:**

Load the CIFAR-10 dataset and normalize image pixel values.

**STEP 3:**

Load the pre-trained MobileNetV2 model (without the top layers) with ImageNet weights.

**STEP 4:**

Freeze the base model layers to retain pre-learned features.

**STEP 5:**

Add new layers for classification — GlobalAveragePooling, Dense, Dropout, and final Softmax.

**STEP 6:**

Compile the model using the Adam optimizer and sparse categorical cross-entropy loss.

**STEP 7:**

Train the model for several epochs and validate using the test dataset.

**STEP 8:**

Evaluate the model’s performance and save the trained model for future use.

## **PROGRAM**

**Name:** S MOHANKUMAR

**Register Number:** 2305002014

```python

# Import Libraries
import tensorflow as tf
from tensorflow.keras import layers, models

# Load Dataset
(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.cifar10.load_data()

# Normalize the Data
train_images, test_images = train_images / 255.0, test_images / 255.0

# Load Pretrained Model and Modify for Transfer Learning
base_model = tf.keras.applications.MobileNetV2(
    input_shape=(32, 32, 3),
    include_top=False,
    weights='imagenet'
)
base_model.trainable = False  # Freeze the base layers

# Build the Model
model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(10, activation='softmax')
])

# Compile the Model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Train the Model
history = model.fit(train_images, train_labels,
                    epochs=5,
                    validation_data=(test_images, test_labels))

# Evaluate the Model
loss, accuracy = model.evaluate(test_images, test_labels)
print(f"\nTest Accuracy: {accuracy*100:.2f}%")

# Save the Model
model.save("Transfer_Learning_Model.h5")
print("\nModel Saved Successfully")
```

## **OUTPUT**

**Epochs Training**

<img width="1227" height="236" alt="image" src="https://github.com/user-attachments/assets/08c529cc-9c0b-43a0-9168-695c72db7b8c" />

---
**Accuracy**

<img width="833" height="50" alt="image" src="https://github.com/user-attachments/assets/e8a7e3bd-4894-41eb-81d1-2170eb29f06a" />

---
**Training Loss, Validation Loss Vs Iteration Plot**

<img width="700" height="547" alt="image" src="https://github.com/user-attachments/assets/c6f7846e-599c-4770-887a-2d3cbb173323" />

---
**Confusion Matrix**

<img width="853" height="766" alt="image" src="https://github.com/user-attachments/assets/4a17c6bc-0a2f-4cde-9243-cef878395968" />

---
**Classification Report**

<img width="631" height="446" alt="image" src="https://github.com/user-attachments/assets/e9e7ac42-01bf-4518-a0fc-cfb4fd30aefc" />


## **RESULT**

The transfer learning-based image classification model was successfully implemented using MobileNetV2 on the CIFAR-10 dataset.
