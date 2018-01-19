#coding:utf-8
from flask import Flask
from CNN_test import model_test
from flask import request
import os
from flask import jsonify
import fire
app = Flask(__name__)
predictor = model_test()

@app.route('/')
def greeting():
    return 'greeting from image_classifier'

@app.route('/image_classifier_one_image', methods=["POST"])
def index():
    image_file = request.files['raw_image']
    save_path = os.path.join(os.path.dirname(__file__), 'saved_image_path', image_file.filename)
    image_file.save(save_path)
    label = predictor.run_one_image(save_path)
    res = {}
    res['ret_status'] = 'success'
    res['ret_data'] = {}
    res['ret_data']['label_index'] = str(label[0])
    if label[0] == 0:
        res['ret_data']['label_discription'] = 'It is not target image, not answer page'
    else:
        res['ret_data']['label_discription'] = 'It is target image, answer page'
    print(res)
    return jsonify(res)

def main(host = "0.0.0.0", port = 5000):
    app.run(host= host, port= port)

if __name__== "__main__":
    fire.Fire()