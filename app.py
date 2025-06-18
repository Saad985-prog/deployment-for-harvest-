from flask import Flask, request, render_template, send_from_directory
from keras.models import load_model
from keras.preprocessing import image
import numpy as np
import os
import uuid

# تحميل النموذج
model = load_model('intel_mobilenet_model.h5')
class_names = ['Buildings', 'Forest', 'Glacier', 'Mountain', 'Sea', 'Street']

# إنشاء مجلد لحفظ الصور
os.makedirs("static", exist_ok=True)

app = Flask(__name__)

def prepare_image(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0
    return img_array

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    image_url = None
    if request.method == "POST":
        file = request.files['file']
        filename = f"{uuid.uuid4().hex}.jpg"
        filepath = os.path.join("static", filename)
        file.save(filepath)

        img = prepare_image(filepath)
        pred = model.predict(img)
        prediction = class_names[np.argmax(pred)]
        image_url = f"/static/{filename}"
    return render_template("index.html", prediction=prediction, image_url=image_url)

@app.route("/static/<path:filename>")
def static_files(filename):
    return send_from_directory("static", filename)

if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
