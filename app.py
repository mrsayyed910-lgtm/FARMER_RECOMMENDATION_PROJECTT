from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

crop_model = pickle.load(open("models/crop_recom_best_model.pkl", "rb"))
crop_encoder = pickle.load(open("models/crop_recom_encoder.pkl", "rb"))
crop_scaler = pickle.load(open("models/crop_recom_scaler.pkl", "rb"))


fertilizer_model = pickle.load(open("models/fertilizer_best_model.pkl", "rb"))
fertilizer_scaler = pickle.load(open("models/fertilizer_scaler.pkl", "rb"))

soil_encoder = pickle.load(open("models/soil_encoder.pkl", "rb"))
crop_name_encoder = pickle.load(open("models/crop_encoder.pkl", "rb"))
fertilizer_encoder = pickle.load(open("models/fertilizer_encoder.pkl", "rb"))


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/crop")
def crop():
    return render_template("crop.html")

@app.route("/predict_crop", methods=["POST"])
def predict_crop():

    N = float(request.form["N"])
    P = float(request.form["P"])
    K = float(request.form["K"])
    temperature = float(request.form["temperature"])
    humidity = float(request.form["humidity"])
    ph = float(request.form["ph"])
    rainfall = float(request.form["rainfall"])

    data = np.array([[N, P, K, temperature, humidity, ph, rainfall]])

    data = crop_scaler.transform(data)

    prediction = crop_model.predict(data)

    crop = crop_encoder.inverse_transform(prediction)

    return render_template(
        "crop.html",
        prediction=crop[0]
    )


@app.route("/fertilizer")
def fertilizer():
    return render_template("fertilizer.html")


@app.route("/predict_fertilizer", methods=["POST"])
def predict_fertilizer():

    temperature = float(request.form["temperature"])
    moisture = float(request.form["moisture"])
    rainfall = float(request.form["rainfall"])
    ph = float(request.form["ph"])

    nitrogen = float(request.form["nitrogen"])
    phosphorous = float(request.form["phosphorous"])
    potassium = float(request.form["potassium"])
    carbon = float(request.form["carbon"])

    soil = request.form["soil"]
    crop = request.form["crop"]

    soil = soil_encoder.transform([soil])[0]
    crop = crop_name_encoder.transform([crop])[0]

    data = np.array([[
        temperature,
        moisture,
        rainfall,
        ph,
        nitrogen,
        phosphorous,
        potassium,
        carbon,
        soil,
        crop
    ]])

    data = fertilizer_scaler.transform(data)

    prediction = fertilizer_model.predict(data)

    fertilizer = fertilizer_encoder.inverse_transform(prediction)

    return render_template(
        "fertilizer.html",
        prediction=fertilizer[0]
    )


if __name__ == "__main__":
    app.run(debug=True)