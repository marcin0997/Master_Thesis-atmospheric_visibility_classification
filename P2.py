import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import numpy as np
from google.colab import drive
import tarfile
import os

drive.mount('/content/drive')
tar = tarfile.open("drive/MyDrive/kopia_Test.tar.xz")
tar.extractall("kopiaTest")

train_data_dir = '/content/kopiaTest'

batch_size = 64
image_size = (224, 224)

train_datagen = ImageDataGenerator(
    rescale=1./255,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest',
    validation_split=0.2
)

validation_datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

train_generator = validation_datagen.flow_from_directory(
    train_data_dir,
    target_size=image_size,
    batch_size=batch_size,
    class_mode='categorical',
    subset='training'
)

validation_generator = validation_datagen.flow_from_directory(
    train_data_dir,
    target_size=image_size,
    batch_size=batch_size,
    class_mode='categorical',
    subset='validation'
)


model = Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(2, activation='softmax')
])

model.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['accuracy'])
early_stopping = EarlyStopping(monitor='val_accuracy', patience=5, restore_best_weights=True)
model_checkpoint = ModelCheckpoint('best_model.keras', save_best_only=True, monitor='val_accuracy')


# Trenowanie modelu
history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // batch_size,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // batch_size,
    epochs=30
)


# Plot the Loss

plt.plot(history.history['loss'], label = 'train_loss')
plt.plot(history.history['val_loss'], label ='val loss')
plt.legend()
plt.show()
# plt.savefig('LossVal_loss')



# Przewidywanie na zbiorze walidacyjnym
Y_pred = model.predict(validation_generator, validation_generator.samples // batch_size + 1)
y_pred = np.argmax(Y_pred, axis=1)

# Prawdziwe etykiety walidacyjne
y_true = validation_generator.classes

# Tworzenie macierzy błędów
cm = confusion_matrix(y_true, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=validation_generator.class_indices.keys())

# Wyświetlenie macierzy błędów
disp.plot(cmap=plt.cm.Blues)
plt.title('Macierz Błędów')
plt.show()

# Ewaluacja modelu na zbiorze walidacyjnym
loss, acc = model.evaluate(validation_generator, steps=validation_generator.samples // batch_size)
print(f"Dokładność modelu: {acc * 100:.2f}%")

model.save('my_modelfinal_9766procent.keras')



#TESTY
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import models, layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator


test_data_gen = ImageDataGenerator(rescale=1./255)


test_generator = test_data_gen.flow_from_directory(
    'ost', 
    target_size=(224, 224), 
    batch_size=64,
    class_mode='categorical',  
    shuffle=False  
)

loaded_model = keras.models.load_model('my_modelfinal_9766procent.keras')


loss, acc = loaded_model.evaluate(test_generator, steps=test_generator.samples // 64)
print(f"Dokładność załadowanego modelu: {acc * 100:.2f}%")

import tarfile

tar_file_path = '/content/good.tar'

with tarfile.open(tar_file_path) as tar:
    tar.extractall('ost')


test_data_gen = ImageDataGenerator(rescale=1./255)

test_generator = test_data_gen.flow_from_directory(
    'ost',  
    target_size=(224, 224), 
    batch_size=64,
    class_mode='categorical',  
    shuffle=False  
)

loss, acc = loaded_model.evaluate(test_generator, steps=test_generator.samples // test_generator.batch_size)
print(f"Dokładność modelu na zbiorze testowym: {acc * 100:.2f}%")

Y_pred = loaded_model.predict(test_generator, steps=test_generator.samples // test_generator.batch_size + 1)

y_pred = np.argmax(Y_pred, axis=1)
y_true = test_generator.classes
	
cm = confusion_matrix(y_true, y_pred)

# Tworzenie macierzy błędów
cm = confusion_matrix(y_true, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=test_generator.class_indices.keys())

# Wyświetlenie macierzy błędów
disp.plot(cmap=plt.cm.Blues)
plt.title('Macierz Błędów dla Zbioru Testowego')
plt.show() 
