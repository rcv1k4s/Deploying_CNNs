import numpy as np
from PIL import Image
from flask import Flask, request, jsonify
from test import load_graph_and_return_session, test_on_image

app = Flask(__name__)

# Load required mean and std for inference
mean_and_std = np.load('train_mean_and_std.npy',allow_pickle=True).item()
mean = mean_and_std['mean']
std = mean_and_std['std']

# Model frozen graph to be used
graph_f = 'model_save_dir_svhn_copy/frozen_model.pb'
sess, input_n, output_n = load_graph_and_return_session(graph_f)

# Flask app definition to receive a data posting from client
@app.route("/im_process", methods=["POST"])
def process_image():
    file = request.files['image']
    image = np.asarray(Image.open(file.stream))
    
    # Generate prediction using the model 
    pred = test_on_image(image,sess,input_n,output_n, mean, std)
    pred_val = np.argmax(pred) + 1

    return jsonify({'msg': 'success', 'prediction':str(pred_val)})


if __name__ == '__main__':
    # Start the server
    app.run(host='0.0.0.0',port=7718,debug=True)
