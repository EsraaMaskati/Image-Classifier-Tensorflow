# Required libraries
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np


#image preprocessing function
def process_image(image):
    t_image = tf.convert_to_tensor(image)
    resized_image = tf.image.resize(t_image,(224,224))
    norm_image = resized_image/255
    np_image = norm_image.numpy()
    return np_image
