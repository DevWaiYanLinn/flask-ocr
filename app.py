from flask import Flask, request, redirect, jsonify
import numpy as np
from flask import render_template
from src.model.predict import test_pipeline

ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}


app = Flask(__name__, template_folder="./src/templates")
app.secret_key = "1234"


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/photo-upload", methods=["POST"])
def upload_file():
    if "file" not in request.files:
        return redirect(request.url)
    file = request.files["file"]
    if file.filename == "":
        return redirect(request.url)
    if file and allowed_file(file.filename):
        image = file.read()
        image_data = np.frombuffer(image, np.uint8)
        results = test_pipeline(image_data)
        return jsonify(results), 200


if __name__ == '__main__':
    app.run(debug=True)
