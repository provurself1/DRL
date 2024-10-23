from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
import numpy as np

app = Flask(__name__)

# Load the trained model (HDF5 format)
model = load_model('model.h5')

# Define action space (modify based on your environment)
action_space = [0, 1, 2, 3]  # Example actions (left, right, up, down)

@app.route('/predict', methods=['POST'])
def predict_action():
    """
    This endpoint receives the current state of the environment
    and returns the action to be taken by the model.
    """
    data = request.json  # Expecting the state as input
    state = np.array(data['state']).reshape(1, -1)  # Reshape the state for prediction
    
    # Get action probabilities from the model
    action_probs = model.predict(state)
    
    # Select the action with the highest probability
    action = np.argmax(action_probs)
    
    return jsonify({
        'action': int(action),
        'action_name': action_space[action]
    })

@app.route('/')
def home():
    return "DQN Agent API is running!"

if __name__ == '__main__':
    app.run(debug=True)
