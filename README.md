

# Celebrity Image Classification using CNN

## Overview

This image classification model is developed to identify images of five celebrities: Lionel Messi, Maria Sharapova, Roger Federer, Serena Williams, and Virat Kohli. The model employs Convolutional Neural Networks (CNNs) for feature extraction and classification. The README provides a comprehensive overview of the model architecture, training pipeline, evaluation metrics, and conclusion.

## Model Architecture

- **Input Shape**: (128, 128, 3) - 128x128 pixels RGB images
- **Convolutional Layers**:
    - Conv2D (32 filters, 3x3 kernel) → ReLU activation → MaxPooling (2x2)
    - Conv2D (64 filters, 3x3 kernel) → ReLU activation → MaxPooling (2x2)
    - Conv2D (128 filters, 3x3 kernel) → ReLU activation → MaxPooling (2x2)
- **Flatten Layer**: Flattens the output for Dense layers.
- **Dense Layers**:
    - Dense (256 units) → ReLU activation
    - Dropout (dropout rate: 0.5)
    - Dense (512 units) → ReLU activation
    - Dense (5 units) → Softmax activation for multi-class classification

The model is trained using a training pipeline that includes data augmentation to increase the diversity of the training dataset. The ImageDataGenerator is employed to perform operations such as rotation, width and height shifts, shear, zoom, and horizontal flips. The dataset is split into training and testing sets, normalized to the range [0, 1], and one-hot encoded for categorical labels. Early stopping is implemented with a patience of 5 epochs, monitoring the training accuracy.


## Data Augmentation

The dataset is augmented by applying random transformations to each image. For each original image, 10 augmented versions are generated using random rotations, shifts, shearing, zooming, and horizontal flips.Augmentation enhances the diversity of the training dataset, contributing to improved model generalization.

## Training Parameters

- **Optimizer**: Adam
- **Loss Function**: Categorical Crossentropy
- **Metrics**: Accuracy
- **Epochs**: 30
- **Batch Size**: 32

## Model Evaluation

The model's performance is evaluated using accuracy and the classification report, which includes precision, recall, and F1-score for each class. The model is trained to minimize categorical crossentropy loss.


## Conclusion

The model achieves good accuracy of 96% on the testing set, showing it's ability to classify celebrity images. Early stopping helps prevent overfitting, and data augmentation contributes to the model's ability to generalize well to unseen data. The classification report provides detailed insights into the model's performance for each celebrity class.

## Usage

The trained model is saved as "celebrity_model.h5" for future use. To make predictions on new images, load the model and utilize the provided `make_prediction` function.

## Dependencies

- Python
- TensorFlow
- NumPy
- OpenCV
- scikit-learn

