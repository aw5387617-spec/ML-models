import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np

MODEL_PATH = "cat_dog_model.h5"
model = tf.keras.models.load_model(MODEL_PATH)
IMAGE_PATH = "dog.jpg"
img = image.load_img(IMAGE_PATH, target_size=(224, 224))
img_array = image.img_to_array(img)
img_array = img_array / 255.0
img_array = np.expand_dims(img_array, axis=0)
prediction = model.predict(img_array)[0][0]
if prediction > 0.5:
    print(f"Prediction: DOG ({prediction:.2f})")
else:
    print(f"Prediction: CAT ({1 - prediction:.2f})")
