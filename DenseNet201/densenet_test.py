import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.densenet import preprocess_input

BEST_MODEL_FILE = "/Users/danielsong/Desktop/DS Project/densenet201_fundus_best.h5"
IMG_SIZE        = (512, 512)
DATA_DIR        = "/Users/danielsong/Desktop/DS Project/Organized_Images"

model = tf.keras.models.load_model(BEST_MODEL_FILE)
print(model.summary())

categories = sorted(os.listdir(DATA_DIR))
print("Categories:", categories)

def prepare_image(img_path):
    img = cv2.imread(img_path)
    img_resized = cv2.resize(img, IMG_SIZE, interpolation=cv2.INTER_AREA)
    batch = np.expand_dims(img_resized, axis=0)
    return preprocess_input(batch), img

# img prediction
test_image_path = "/Users/danielsong/Desktop/DS Project/Organized_Images/Class_2/5_right.jpg"
input_batch, orig_img = prepare_image(test_image_path)

# run the model
probs = model.predict(input_batch, verbose=0)[0]
pred_idx = np.argmax(probs)
pred_class = categories[pred_idx]
pred_conf  = probs[pred_idx]

print(f"Predicted: {pred_class} ({pred_conf*100:.2f}%)")

font = cv2.FONT_HERSHEY_COMPLEX
cv2.putText(orig_img,
            f"{pred_class}: {pred_conf*100:.1f}%",
            (20, 40),
            font, 1,
            (0, 255, 255), 2)

cv2.imshow("Prediction", orig_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
