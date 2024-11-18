from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np

# Load the trained model
rfc  = pickle.load(open('model.pkl','rb'))
ms = pickle.load(open('minmaxscaler.pkl','rb'))

crops_dict = {
    "rice": 1,
    "maize": 2,
    "chickpea": 3,
    "kidneybeans": 4,
    "pigeonpeas": 5,
    "mothbeans": 6,
    "mungbean": 7,
    "blackgram": 8,
    "lentil": 9,
    "pomegranate": 10,
    "banana": 11,
    "mango": 12,
    "grapes": 13,
    "watermelon": 14,
    "muskmelon": 15,
    "apple": 16,
    "orange": 17,
    "papaya": 18,
    "coconut": 19,
    "cotton": 20,
    "jute": 21,
    "coffee": 22,
}

def recommend(N, P, k, temperature, humidity, ph, rainfall):
    features = np.array([[N, P, k, temperature, humidity, ph, rainfall]])
    transformed_features = ms.transform(features) 
    prediction = rfc.predict(transformed_features)
    return prediction[0]

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/recommend', methods=['POST'])
def recommend_crop():
    try:
        # Extract form data
        data = request.form
        N = float(data.get('N'))
        P = float(data.get('P'))
        K = float(data.get('K'))
        temperature = float(data.get('temperature'))
        humidity = float(data.get('humidity'))
        ph = float(data.get('ph'))
        rainfall = float(data.get('rainfall'))

        # Make prediction
        predicted_crop = recommend(N, P, K, temperature, humidity, ph, rainfall)
        
        # Map prediction to crop name
        crop_dict2 = {v: k for k, v in crops_dict.items()}
        if predicted_crop in crop_dict2:
            crop_name = crop_dict2[predicted_crop]
        else:
            crop_name = "Sorry, we are unable to recommend the crop now!"
    except Exception as e:
        crop_name = f"Error occurred: {str(e)}"

    return render_template('index.html', crop_name=crop_name)

if __name__ == "__main__":
    app.run(debug=True)