from flask import Flask, request, jsonify
from flask_cors import CORS, cross_origin
import numpy as np
from tensorflow.keras import models
from train import predictShot


app = Flask(__name__)
cors = CORS(app, resources={r"/*": {"origins": "*"}})

model = models.load_model('trained_model')

@app.route('/predict', methods=['POST'])
def predict():
  content_type = request.headers.get('Content-Type')
  json = request.json
  state = np.array(json['state'])

  y, x, probs = predictShot(state, model)

  return jsonify({'x': x, 'y': y, 'probabilities': probs})
