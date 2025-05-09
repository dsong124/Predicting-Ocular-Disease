import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.densenet import preprocess_input
from sklearn.metrics import f1_score, classification_report

DATA_DIR    = 'Organized_Images'
IMG_HEIGHT  = 512
IMG_WIDTH   = 512
BATCH_SIZE  = 8
VALID_SPLIT = 0.2
SEED        = 42
AUTOTUNE    = tf.data.AUTOTUNE

model = load_model("densenet201_fundus_best.h5")

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
val_ds = raw_val.map(
    lambda x, y: (preprocess_input(x), y),
    num_parallel_calls=AUTOTUNE
).prefetch(AUTOTUNE)

y_true, y_pred = [], []
for batch_images, batch_labels in val_ds:
    probs = model.predict(batch_images, verbose=0)
    y_true.extend(np.argmax(batch_labels.numpy(), axis=1))
    y_pred.extend(np.argmax(probs, axis=1))

y_true = np.array(y_true)
y_pred = np.array(y_pred)

wf1 = f1_score(y_true, y_pred, average='weighted')
print(f"Weighted F1 Score: {wf1:.4f}\n")

print("Per-class precision / recall / F1:\n")
print(classification_report(
    y_true,
    y_pred,
    target_names=raw_val.class_names,
    digits=4
))
