from flask import Flask, request, render_template
from tensorflow.keras.models import load_model # type: ignore
from tensorflow.keras.preprocessing.image import img_to_array, load_img # type: ignore
import numpy as np
import io

app = Flask(__name__)
model = load_model('pokemon_classifier.h5')

def prepare_image(image, target):
    if image.mode != 'RGB':
        image = image.convert('RGB')
    image = image.resize(target)
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    image /= 255.0
    return image

@app.route("/", methods=["GET", "POST"])
def upload_predict():
    if request.method == "POST":
        image_file = request.files["image"]
        if image_file:
            image = load_img(io.BytesIO(image_file.read()), target_size=(224, 224))
            image = prepare_image(image, target=(224, 224))
            preds = model.predict(image)
            result = "Pokemon" if preds[0] > 0.5 else "Not a Pokemon"
            return render_template("index.html", result=result)
    return render_template("index.html", result=None)

if __name__ == "__main__":
    app.run(debug=True)
