import json
from io import BytesIO
import base64

from flask import Flask, render_template, jsonify
import numpy as np
import tensorflow as tf
import keras
import matplotlib.pyplot as plt



app = Flask(__name__)

def load_model(dataset_name):
    """
    Loading  of the dataset.
    """
    if dataset_name not in {'cwru', 'mfd', 'tri', 'transfer'}:
        raise ValueError("Invalid dataset name. Choose from 'cwru', 'mfd', 'tri' or 'transfer'.")
    
    model = tf.keras.models.load_model(f'models/{dataset_name}_model.h5')

    return model

def unpk_dataset(dataset_name):
    """
    Unpacking variables.
    """
    if dataset_name not in {'cwru', 'mfd', 'tri'}:
        raise ValueError("Invalid dataset name. Choose from 'cwru', 'mfd', 'tri'.")
            
    data = np.load(f'models/{dataset_name}_samples.npz')
    encfile = open(f'models/{dataset_name}_encoding.json', 'r', encoding='utf-8')
    encord = json.load(encfile)
    encfile.close()
    
    smpl_num = data['smpl_num'].tolist()
    xsmpl = data['xsmpl'].tolist()
    ysmpl = data['ysmpl'].tolist()
       
    return encord, smpl_num, xsmpl, ysmpl

@app.route('/')
def index():
    """
    Render the main page.
    """
    return render_template('index.html')

@app.route('/cwru')
def cwru():
    """
    Render the cwru page.
    """
    enc_ord, smpl_num, _, ysmpl = unpk_dataset('cwru')
    dec_lbl = [enc_ord[label] for label in np.argmax(ysmpl, axis=1).tolist()]
    samples = zip(smpl_num, dec_lbl)

    return render_template('cwru.html', samples=samples)

@app.route('/mfd')
def mfd():
    """
    Render the cwru page.
    """
    enc_ord, smpl_num, _, ysmpl = unpk_dataset('mfd')
    dec_lbl = [enc_ord[label] for label in np.argmax(ysmpl, axis=1).tolist()]
    samples = zip(smpl_num, dec_lbl)
    

    return render_template('mfd.html', samples=samples)

@app.route('/tri')
def tri():
    """
    Render the cwru page.
    """
    enc_ord, smpl_num, _, ysmpl =unpk_dataset('tri')
    dec_lbl = [enc_ord[label] for label in np.argmax(ysmpl, axis=1).tolist()]
    samples = zip(smpl_num, dec_lbl)

    return render_template('tri.html', samples=samples)

@app.route('/transfer')
def transfer():
    """
    Render the transfer page.
    """
    enc_ord, smpl_num, _, ysmpl =unpk_dataset('tri')
    dec_lbl = [enc_ord[label] for label in np.argmax(ysmpl, axis=1).tolist()]
    samples = zip(smpl_num, dec_lbl)

    return render_template('transfer.html', samples=samples)

@app.route('/predict/<string:dataset>/<string:smpl_num>')
def predict(dataset, smpl_num):
    """
    Render the prediction page.
    """
    try:
        smpl = int(smpl_num)

        model = load_model(dataset)

        if dataset == 'transfer':
            dataset = 'tri'

        enc_ord, smpl_num, xsmpl, ysmpl = unpk_dataset(dataset)

        idx = smpl_num.index(smpl)
        x = xsmpl[idx]
        y = ysmpl[idx]
        x = np.expand_dims(x, axis=0)

        pred = model.predict(x)
        pred = enc_ord[np.argmax(pred)]
        label = enc_ord[np.argmax(y)]

        buf = BytesIO()
        plt.plot(x[0])
        plt.title('Signal Plot')
        plt.xlabel('Time (s)')
        plt.ylabel('Amplitude')
        plt.savefig(buf, format='png')
        buf.seek(0)
        plt.close()
        img_str = base64.b64encode(buf.getvalue()).decode('utf-8')
        buf.close()
        
        return jsonify({
            'prediction': pred,
            'label': label,
            'plot': img_str
        })
    except ValueError as e:
        return jsonify({
            'error': str(e),
            'message': 'Invalid sample number format'
        }), 400
    except Exception as e:
        return jsonify({
            'error': str(e),
            'message': 'An error occurred while processing the request.'
        }), 500

# This is the command for the development server we've moved up the leagues darling gunicorn is handling serving this app now
if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5001, debug=True)
