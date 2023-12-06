import numpy as np
import cv2
import os
import tensorflow as tf
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from keras.utils import to_categorical
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import EarlyStopping

from tqdm import tqdm
from keras.utils import to_categorical

image_dir='cropped'
lionel_messi = os.listdir(image_dir+ '/lionel_messi')
maria_sharapova = os.listdir(image_dir+ '/maria_sharapova')
roger_federer = os.listdir(image_dir+ '/roger_federer')
serena_williams = os.listdir(image_dir+ '/serena_williams')
virat_kohli = os.listdir(image_dir+ '/virat_kohli')
print("--------------------------------------\n")

dataset=[]
label=[]
img_siz=(128,128)


for i , image_name in tqdm(enumerate(lionel_messi),desc="Lionel Messi"):
    if(image_name.split('.')[1]=='png'):
        image=cv2.imread(image_dir+'/lionel_messi/'+image_name)
        image=Image.fromarray(image,'RGB')
        image=image.resize(img_siz)
        dataset.append(np.array(image))
        label.append(0)
        
        
for i ,image_name in tqdm(enumerate(maria_sharapova),desc="Maria Sharapova"):
    if(image_name.split('.')[1]=='png'):
        image=cv2.imread(image_dir+'/maria_sharapova/'+image_name)
        image=Image.fromarray(image,'RGB')
        image=image.resize(img_siz)
        dataset.append(np.array(image))
        label.append(1)

for i , image_name in tqdm(enumerate(roger_federer),desc="Roger Federer"):
    if(image_name.split('.')[1]=='png'):
        image=cv2.imread(image_dir+'/roger_federer/'+image_name)
        image=Image.fromarray(image,'RGB')
        image=image.resize(img_siz)
        dataset.append(np.array(image))
        label.append(2)
for i , image_name in tqdm(enumerate(serena_williams),desc="Serena Williams"):
    if(image_name.split('.')[1]=='png'):
        image=cv2.imread(image_dir+'/serena_williams/'+image_name)
        image=Image.fromarray(image,'RGB')
        image=image.resize(img_siz)
        dataset.append(np.array(image))
        label.append(3)
for i , image_name in tqdm(enumerate(virat_kohli),desc="Virat Kohli"):
    if(image_name.split('.')[1]=='png'):
        image=cv2.imread(image_dir+'/virat_kohli/'+image_name)
        image=Image.fromarray(image,'RGB')
        image=image.resize(img_siz)
        dataset.append(np.array(image))
        label.append(4)
        
dataset=np.array(dataset)
label = np.array(label)

print("--------------------------------------\n")
print('Dataset Length: ',len(dataset))
print('Label Length: ',len(label))
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
# Data augmentation
datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

# Early stopping
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# Model definition
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense(5, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Training with data augmentation and early stopping
print("--------------------------------------\n")
print("Training Started.\n")
history = model.fit(
    datagen.flow(x_train, y_train_encoded, batch_size=32),
    epochs=50,
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

# Classification report
y_pred = model.predict(x_test)
y_pred_classes = np.argmax(y_pred, axis=1)
print('Classification Report\n', classification_report(y_test, y_pred_classes, target_names=class_names))
print("--------------------------------------\n")

# Model Prediction
print("--------------------------------------\n")
print("Model Prediction.\n")
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

print(make_prediction('cropped/fed_test.png', model))
print("--------------------------------------\n")
