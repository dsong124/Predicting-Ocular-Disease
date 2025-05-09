import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.densenet import preprocess_input
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

DATA_DIR    = 'Organized_Images'
IMG_HEIGHT  = 512
IMG_WIDTH   = 512
BATCH_SIZE  = 8
VALID_SPLIT = 0.2
SEED        = 42
AUTOTUNE    = tf.data.AUTOTUNE

model = load_model('densenet201_fundus_best.h5')

raw_val = tf.keras.utils.image_dataset_from_directory(
    DATA_DIR,
    labels='inferred',
    label_mode='categorical',
    validation_split=VALID_SPLIT,
    subset='validation',
    seed=SEED,
    image_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE
)
val_ds = raw_val.map(lambda x, y: (preprocess_input(x), y), num_parallel_calls=AUTOTUNE).prefetch(AUTOTUNE)

y_true, y_pred = [], []
for images, labels in val_ds:
    probs = model.predict(images, verbose=0)
    y_true.extend(np.argmax(labels.numpy(), axis=1))
    y_pred.extend(np.argmax(probs, axis=1))

y_true = np.array(y_true)
y_pred = np.array(y_pred)

cm = confusion_matrix(y_true, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=raw_val.class_names)

plt.figure(figsize=(8, 8))
disp.plot(cmap='Blues', xticks_rotation='vertical')
plt.title('Confusion Matrix: True vs Predicted Classes')
plt.tight_layout()
plt.show()