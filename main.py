# Example main script to run all steps

# Import necessary functions from other scripts
from model_training import *
from model_evaluation import *
from predict import predict_image

# Define the test image path
test_image_path = 'path/to/test/image.jpg'  # Update this path to your test image

# Predict the test image
print(predict_image(test_image_path, model))
