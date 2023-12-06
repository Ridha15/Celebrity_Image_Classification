import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
import tensorflow as tf
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from tqdm import tqdm
from keras.utils import to_categorical

# function to load and process images
def load_and_preprocess_image(img_path):
    image = cv2.imread(img_path)
    image = Image.fromarray(image, 'RGB')
    image = image.resize(img_siz)
    return np.array(image)

# Data augmentation using ImageDataGenerator
datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

image_dir = 'cropped'
class_directories = {
    'lionel_messi': 0,
    'maria_sharapova': 1,
    'roger_federer': 2,
    'serena_williams': 3,
    'virat_kohli': 4
}

dataset = []
label = []
img_siz = (128, 128)

for class_name, class_label in class_directories.items():
    images = os.listdir(os.path.join(image_dir, class_name))
    for i, image_name in tqdm(enumerate(images), desc=class_name.capitalize()):
        if image_name.split('.')[-1].lower() == 'png':
            img_path = os.path.join(image_dir, class_name, image_name)
            original_image = load_and_preprocess_image(img_path)
            # augmenting each image 10 times
            for _ in range(10):  
                augmented_image = datagen.random_transform(original_image)
                dataset.append(augmented_image)
                label.append(class_label)

dataset = np.array(dataset)
label = np.array(label)

print("--------------------------------------\n")
print('Augmented Dataset Length: ', len(dataset))
print('Augmented Label Length: ', len(label))
print("--------------------------------------\n")



print("--------------------------------------\n")
print("Train-Test Split")
x_train,x_test,y_train,y_test=train_test_split(dataset,label,test_size=0.2,random_state=42)
print("--------------------------------------\n")

print("--------------------------------------\n")
print("Normalaising the Dataset. \n")
x_train=x_train.astype('float')/255
x_test=x_test.astype('float')/255 

y_train_encoded = to_categorical(y_train, num_classes=5)
y_test_encoded = to_categorical(y_test, num_classes=5)
class_names = ['Lionel Messi', 'Maria Sharapova', 'Roger Federer', 'Serena Williams', 'Virat Kohli']

# model architecture
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense(5, activation='softmax')
])

early_stopping = tf.keras.callbacks.EarlyStopping(monitor='accuracy', patience=5, restore_best_weights=True)

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

print("--------------------------------------\n")
print("Training Started.\n")
history = model.fit(
    datagen.flow(x_train, y_train_encoded, batch_size=32),
    epochs=30,
    validation_data=(x_test, y_test_encoded),
    callbacks=[early_stopping]
)
print("Training Finished.\n")
print("--------------------------------------\n")

# Model evaluation
print("--------------------------------------\n")
print("Model Evaluation Phase.\n")
loss, accuracy = model.evaluate(x_test, y_test_encoded)
print(f'Accuracy: {round(accuracy*100, 2)}')

y_pred = model.predict(x_test)
y_pred_classes = np.argmax(y_pred, axis=1)
print('Classification Report\n', classification_report(y_test, y_pred_classes, target_names=class_names))
print("--------------------------------------\n")

#saving the model
model.save("celebrity_model.h5")

# loading the model for predictions
saved_model = tf.keras.models.load_model("celebrity_model.h5")

print("--------------------------------------\n")
print("Model Prediction.\n")
# function to make predictions on new images
def make_prediction(img, model):
    img = cv2.imread(img)
    img = Image.fromarray(img, 'RGB')
    img = img.resize((128, 128))
    img = np.array(img)
    input_img = np.expand_dims(img, axis=0)
    predictions = model.predict(input_img)
    predicted_class = np.argmax(predictions, axis=1)[0]
    predicted_class_name = class_names[predicted_class]
    return predicted_class_name

# predictions on new images
print(make_prediction('cropped/fed_test.png', saved_model))
print(make_prediction('cropped/kohli_test.jpeg', saved_model))
print(make_prediction('cropped/maria_test.jpg', saved_model))
print(make_prediction('cropped/messi_test.jpeg', saved_model))
print(make_prediction('cropped/serena_test.jpg', saved_model))
print("--------------------------------------\n")
