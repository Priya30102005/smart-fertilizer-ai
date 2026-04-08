from flask import Flask, request, render_template
import pickle

app = Flask(__name__)

# Load AI model (Agent brain)
model = pickle.load(open("model.pkl", "rb"))
le_soil = pickle.load(open("le_soil.pkl", "rb"))
le_crop = pickle.load(open("le_crop.pkl", "rb"))
le_fert = pickle.load(open("le_fert.pkl", "rb"))

# Home page
@app.route('/')
def home():
    return render_template("index.html")

# AI Agent decision
@app.route('/predict', methods=['POST'])
def predict():

    # Get values safely
    temp = request.form.get("Temperature")
    humidity = request.form.get("Humidity")
    moisture = request.form.get("Moisture")
    n = request.form.get("N")
    p = request.form.get("P")
    k = request.form.get("K")

    soil = request.form.get("Soil_Type")
    crop = request.form.get("Crop_Type")

    # Default values
    temp = float(temp) if temp else 28
    humidity = float(humidity) if humidity else 60
    moisture = float(moisture) if moisture else 40
    n = float(n) if n else 50
    p = float(p) if p else 40
    k = float(k) if k else 30

    # Encode
    soil_enc = le_soil.transform([soil])[0]
    crop_enc = le_crop.transform([crop])[0]

    # Input
    input_data = [[temp, humidity, moisture, soil_enc, crop_enc, n, p, k]]

    # Predict
    prediction = model.predict(input_data)
    result = le_fert.inverse_transform(prediction)[0]

    return render_template("index.html", prediction=result)
if __name__ == "__main__":
    app.run(debug=True)