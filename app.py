import os
import json
import logging
from PIL import Image
from flask_cors import CORS
from flask_restful import Api
from keras.models import load_model
from numpy import argmax, expand_dims
from flask import Flask, request,flash
from flasgger import Swagger, swag_from
from tensorflow.keras.preprocessing.image import load_img, img_to_array
# from keras.preprocessing.image import image



UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

FILENAME = ''

app = Flask(__name__)
CORS(app)
api = Api(app)
swagger = Swagger(app)

def preprocess_image(img):
  img = img.resize((600, 600))
  img = image.img_to_array(img)
  img = expand_dims(img, axis=0)
  img = img / 255.0
  return img

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/upload', methods=['POST'])
@swag_from({
        'consumes': ['multipart/form-data'],
        'parameters': [
            {
                'name': 'file',
                'in': 'formData',
                'type': 'file',
                'required': True,
                'description': 'The file to be uploaded'
            }
        ],
        'responses': {
            200: {
                'description': 'File details',
                'schema': {
                    'type': 'object',
                    'properties': {
                        'filename': {'type': 'string'},
                        'content_type': {'type': 'string'},
                        'size': {'type': 'integer'}
                    }
                }
            },
            400: {
                'description': 'Invalid input'
            }
        }
    })
def upload_file():

    file = request.files['file']
    if file.filename == '':
        flash("No selected file")
        return json.dumps({'error': 'Error processing the image'}, indent=4)
    
    if file and allowed_file(file.filename):
        try:
            global FILENAME
            FILENAME = file.filename
            
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], FILENAME)
            logging.info(f"Saving file to {file_path}")
            file.save(file_path)
            
            file_details = {
                'filename': file.filename,
                'content_type': file.content_type,
                'size': os.path.getsize(file_path)
            }
            return json.dumps({"Uploaded File Info ": file_details}, indent=4)
        except Exception as e:
            logging.error(f"Error processing the image: {e}")
            return json.dumps({"Error": str(e)},indent=4)

    return json.dumps({"error": "failed"},indent=4)

@app.route('/predict', methods=['GET'])
@swag_from({
    'responses': {
        200: {
            'description': 'Model Evaluation',
            'schema': {
                'type': 'object',
                'properties': {
                    'Model Evaluation': {
                        'type': 'object',
                        'properties': {
                            'Mapped Class': {'type': 'string'},
                            'Predicted Index': {'type': 'string'}
                        }
                    }
                }
            }
        },
        500: {
            'description': 'Error processing the image',
            'schema': {
                'type': 'object',
                'properties': {
                    'Error': {'type': 'string'}
                }
            }
        }
    }
})
def prediction():
    try:
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], FILENAME)
        logging.info(f"Loading file from from:  {file_path}")
        image = Image.open(file_path)
        current_dir = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(current_dir, 'model_checkpoint_6.h5')
        model = load_model(model_path,compile=False)
        preprocessed_img = preprocess_image(image)
        prediction = model.predict(preprocessed_img)
        predicted_class = argmax(prediction, axis=1)
        mappings = { 0: 'Finger millet', 1: 'Pearl millet' }
        data = {
            'Mapped Class': mappings[predicted_class[0]],
            'Predicted Index': str(predicted_class[0])
        }
        os.remove(file_path)
        return json.dumps({"Model Evaluation ": data})
    except Exception as e:
        logging.error(f"Error processing the image: {e}")
        return json.dumps({"Error": str(e)}, indent=4)

@app.route('/predict/<filename>', methods=['GET'])
@swag_from({
    'parameters': [
        {
            'name': 'filename',
            'in': 'path',
            'type': 'string',
            'required': True,
            'description': 'Name of the file to be predicted'
        }
    ],
    'responses': {
        200: {
            'description': 'Model Evaluation',
            'schema': {
                'type': 'object',
                'properties': {
                    'Model Evaluation': {
                        'type': 'object',
                        'properties': {
                            'Mapped Class': {'type': 'string'},
                            'Predicted Index': {'type': 'string'}
                        }
                    }
                }
            }
        },
        500: {
            'description': 'Error processing the image',
            'schema': {
                'type': 'object',
                'properties': {
                    'Error': {'type': 'string'}
                }
            }
        }
    }
})
def predict(filename):
    try:
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        logging.info(f"Loading file from: {file_path}")
        image = Image.open(file_path)
        current_dir = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(current_dir, 'model_checkpoint_6.h5')
        model = load_model(model_path, compile=False)
        preprocessed_img = preprocess_image(image)
        prediction = model.predict(preprocessed_img)
        predicted_class = argmax(prediction, axis=1)
        mappings = {0: 'Finger millet', 1: 'Pearl millet'}
        data = {
            'Mapped Class': mappings[predicted_class[0]],
            'Predicted Index': str(predicted_class[0])
        }
        os.remove(file_path)
        return json.dumps({"Model Evaluation": data}, indent=4)
    except Exception as e:
        logging.error(f"Error processing the image: {e}")
        return json.dumps({"Error": str(e)},indent=4)

@app.route('/cleanup', methods=["DELETE"])
@swag_from({
    'responses': {
        200: {
            'description': 'Clean up application upload directory...'
        },
    }
})
def cleanup():
    for file in os.listdir('./uploads'):
        os.remove(os.path.join('./uploads',file))  
    return json.dumps({"clean up": 'Resource clean up...'},indent=4)
    
if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    app.run(debug=True)
