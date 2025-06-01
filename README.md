#  Dog Image Classification with TensorFlow

This project is a deep learning-based image classifier that identifies dog breeds from images using a Convolutional Neural Network (CNN) implemented in TensorFlow and Keras.

---

##  Table of Contents

* [About the Project](#about-the-project)
* [Dataset](#dataset)
* [Model Architecture](#model-architecture)
* [Data Augmentation](#data-augmentation)
* [Training](#training)
* [Evaluation](#evaluation)
* [Results](#results)
* [Installation](#installation)
* [Usage](#usage)
* [Future Improvements](#future-improvements)
* [License](#license)

---

##  About the Project

The goal of this project is to build an image classification model that can recognize and classify different dog breeds from images. It demonstrates how to preprocess image data, build a CNN, apply data augmentation, and evaluate performance.

---

##  Dataset

* The dataset consists of dog images categorized into multiple breeds.
* All images are resized to **256x256** pixels.
* Dataset is split into:

  * **Training set**
  * **Validation/Test set**

Data

The Dataset for the project is given below for download:

**Download link:** [dog_images](https://drive.google.com/your-shared-link)

Note: The original dataset is from Stanford 

The data is expected to be organized like:

```
images/
├── beagle/
├── german_shepherd/
├── golden_retriever/
├── bulldog/
└── poodle/
```


---

##  Model Architecture

The model is a Sequential CNN with the following layers:

```python
network = [
  tf.keras.layers.Rescaling(1./255),
  layers.Conv2D(16, 4, padding='same', activation='relu', input_shape=(256,256,3)),
  layers.MaxPooling2D(),
  layers.Conv2D(32, 4, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Conv2D(64, 4, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Dropout(0.2),
  layers.Flatten(),
  layers.Dense(128, activation='relu'),
  layers.Dense(len(breeds))
]
```

* `Rescaling`: Normalizes pixel values
* `Conv2D` + `MaxPooling2D`: Feature extraction
* `Dropout`: Regularization to prevent overfitting
* `Dense`: Fully connected layers

---

##  Data Augmentation

To increase model generalization and prevent overfitting, data augmentation is applied:

```python
data_augmentation = tf.keras.Sequential([
  layers.RandomFlip("horizontal", seed=1),
  layers.RandomRotation(0.2, seed=1),
  layers.RandomZoom(0.2, seed=1),
])
```

Augmentation is applied **during training**, so the model trains on a transformed version of the input image each time.

---

##  Training

The model is compiled and trained with:

```python
model.compile(
  optimizer='adam',
  loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
  metrics=['accuracy']
)
```

* **Optimizer**: Adam (adaptive learning rate and momentum)
* **Loss**: Categorical cross-entropy (multi-class classification)
* **Metrics**: Accuracy

Model is trained using:

```python
history = model.fit(train, validation_data=test, epochs=5, verbose=1)
```

---

##  Evaluation

Post-training performance is evaluated using:

* Accuracy plots
* Loss curves
* Classification report

```python
import pandas as pd
history_df = pd.DataFrame.from_dict(history.history)
history_df[["accuracy", "val_accuracy"]].plot()
```

---

##  Results

* Sample predictions and actual images shown
* Training vs Validation accuracy plotted

---


##  Future Improvements

* Use a pre-trained model (e.g., ResNet50, MobileNetV2) for better performance
* Add Grad-CAM for visual explanation
* Build a simple web interface to upload and classify images
* Hyperparameter tuning using validation performance

---

##  License

MIT License. See `LICENSE` file for details.
