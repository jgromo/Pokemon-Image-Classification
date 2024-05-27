from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np

# Image parameters
img_height, img_width = 224, 224

# Load the trained model
model = load_model('pokemon_classifier.h5')

def predict_image(img_path, model):
    img = image.load_img(img_path, target_size=(img_height, img_width))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0
    prediction = model.predict(img_array)
    return 'Pokemon' if prediction[0][0] > 0.5 else 'Not a Pokemon'

# Test the function
test_image_path = 'eevee.png'  
print(predict_image(test_image_path, model))
