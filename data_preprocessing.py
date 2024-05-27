import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator # type: ignore

# Define the paths for the input directories
base_dir = 'Images/' 

# Image parameters
img_height, img_width = 224, 224
batch_size = 32

# Create ImageDataGenerators
datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2  # 20% validation data
)

# Create generators
train_generator = datagen.flow_from_directory(
    directory=base_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='binary',
    subset='training'
)

validation_generator = datagen.flow_from_directory(
    directory=base_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='binary',
    subset='validation'
)
