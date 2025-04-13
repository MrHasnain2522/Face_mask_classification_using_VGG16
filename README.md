# Project title:
 😷 Face Mask Classification using VGG16

 # intro:
    The COVID-19 pandemic has emphasized the importance of face masks in reducing the spread of infectious diseases.
    In public spaces like airports, shopping malls, schools, and offices, automated systems that can detect whether ,
    a person is wearing a face mask are becoming increasingly essential for ensuring safety and compliance with health guidelines.

    
    This project aims to build an intelligent image classification system that can automatically distinguish between individuals,
    wearing a face mask and those not wearing one. To achieve this, we leverage transfer learning using the powerful VGG16 convolutional ,
    neural network, which has been pre-trained on the large-scale ImageNet dataset.
    VGG16 is a well-known deep learning model recognized for its simplicity and strong performance on image-related tasks.
    
    Rather than training a convolutional neural network from scratch—which would require a massive dataset and computing,
    resources—we use the pre-trained VGG16 model as a feature extractor and build a custom classification head tailored to,
    our binary classification task.
    By fine-tuning the model on a labeled dataset of masked and unmasked faces, the system learns to detect subtle visual,
    differences and make accurate predictions. This solution can be integrated into real-time video surveillance,
    mobile applications, or embedded systems to promote public health and safety.

## 🧼 🔄 Preprocessing

Before feeding images into the model, they undergo the following preprocessing steps:

- Resize images to **256x256** pixels (VGG16 input size)
- Convert images to arrays and normalize pixel values to the `[0, 1]` range
- Use **ImageDataGenerator** for:
  - Rescaling
  - Augmentation (rotation, zoom, horizontal flip)
- One-hot encoding for labels (binary classification)

Example:
```python
from tensorflow.keras.preprocessing.image import ImageDataGenerator

train_gen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    zoom_range=0.2,
    horizontal_flip=True
)
train_data = train_gen.flow_from_directory(
    'dataset/train',
    target_size=(256, 256),
    class_mode='binary',
    batch_size=32
)


##  🧠 Model Architecture:
- **Base Model:** VGG16 (pre-trained, `include_top=False`)
- **Added Layers:**
  - `GlobalAveragePooling2D`
  - `Dense(128, activation='relu')`
  - `Dropout(0.5)`
  - `Dense(1, activation='sigmoid')`
- **Loss Function:** Binary Crossentropy
- **Optimizer:** Adam
- **Metrics:** Accuracy

## 📁 Dataset

You can use the [Face Mask Detection Dataset on Kaggle](https://www.kaggle.com/datasets/omkargurav/face-mask-dataset)


# Requirements
Python 3.7+
TensorFlow / Keras
NumPy
Matplotlib
scikit-learn

# 📈 Results
Validation Accuracy: ~95%
Fast training with transfer learning
Model saved as mask_detector.h5


   
