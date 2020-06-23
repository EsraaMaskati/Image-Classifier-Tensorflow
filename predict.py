# Required libraries
import argparse
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
from PIL import Image
import json
# Importinf user defined module function
from process_image import process_image



#Define parser object
parser = argparse.ArgumentParser(description="Predict Flower Classification")
#Adding parser arguments
#positional argument: parser.add_argument('path', help="Path of an image to predict")
#optional arguments
parser.add_argument('-img','--image_path', default='./test_images/cautleya_spicata.jpg', help="Path of an image to predict")
parser.add_argument('--model', type=str, default='best_model.h5', help='Model path')
parser.add_argument('--topk', default=5, type=int, help='Defult top_k result')
parser.add_argument('--classes', default='label_map.json', help='class')
#Returning values
arguments = parser.parse_args()
img = arguments.image_path
model = arguments.model
topk = arguments.topk
classes = arguments.classes



#load best_model.h5 model file
model = tf.keras.models.load_model(model,custom_objects={'KerasLayer':hub.KerasLayer})
#read class_names from label_map json file:
with open(classes,'r') as f:
    class_names = json.load(f)

    
    



#image prediction
def predict(image_path, prediction_model , top_k):
  
#1:import image, 2:convert it to numpy, 3,4:resize+normalize+redimnsionimage, 5:convert to list(tensor)
    image = Image.open(image_path)
    img_example = np.asarray(image)    
    processed_image= process_image(img_example)   
    redim_image=np.expand_dims(processed_image,axis=0)
    pred_probability = prediction_model.predict(redim_image)
    pred_probability = pred_probability.tolist()

    #1: get top k valuse and classes, 2:covetr top k valuse and classes to list
    values, indices = tf.math.top_k(pred_probability,k=top_k)
    probs = values.numpy().tolist()[0]
    classes = indices.numpy().tolist()[0]
    label_names= [class_names[str(names+1)] for names in classes]
    print('Prediction result:')
    print('Flower type:\n',label_names)
    print('probabilities: \n',probs,'\n')
    
    
#mapping names
if __name__ == '__main__':
    predict(img, model, topk)
    print('Image: ',img)
    print('model: ',model)
    print('topk: ',topk)
    