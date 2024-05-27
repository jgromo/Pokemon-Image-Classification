from tensorflow.keras.models import load_model
from data_preprocessing import validation_generator

# Load the trained model
model = load_model('pokemon_classifier.h5')

# Evaluate the model
val_loss, val_accuracy = model.evaluate(validation_generator, steps=validation_generator.samples // validation_generator.batch_size)
print(f'Validation Loss: {val_loss}')
print(f'Validation Accuracy: {val_accuracy}')
