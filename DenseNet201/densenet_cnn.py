import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model, Input, optimizers, callbacks
from tensorflow.keras.applications import DenseNet201
from tensorflow.keras.applications.densenet import preprocess_input
from sklearn.utils.class_weight import compute_class_weight

DATA_DIR        = 'Organized_Images'
IMG_HEIGHT      = 512
IMG_WIDTH       = 512
BATCH_SIZE      = 8
VALID_SPLIT     = 0.2
SEED            = 42

EPOCHS_P1       = 8
LR_P1           = 5e-4      
EPOCHS_P2       = 12
LR_P2           = 5e-6     
UNFREEZE_LAYERS = 40

AUTOTUNE = tf.data.AUTOTUNE

# loading up the data/preprocess
raw_train = tf.keras.utils.image_dataset_from_directory(
    DATA_DIR, labels='inferred', label_mode='categorical',
    validation_split=VALID_SPLIT, subset='training',
    seed=SEED, image_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE
)
raw_val = tf.keras.utils.image_dataset_from_directory(
    DATA_DIR, labels='inferred', label_mode='categorical',
    validation_split=VALID_SPLIT, subset='validation',
    seed=SEED, image_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE
)

class_names = raw_train.class_names
num_classes = len(class_names)
print("Classes:", class_names)

# computing class‐weights for Phase 2
y = np.concatenate([y for _, y in raw_train], axis=0)
y_idx = np.argmax(y, axis=1)
cw = compute_class_weight('balanced', classes=np.arange(num_classes), y=y_idx)
class_weight_dict = dict(enumerate(cw))


train_ds_pp = raw_train.map(lambda x, y: (preprocess_input(x), y), num_parallel_calls=AUTOTUNE)
val_ds_pp   = raw_val  .map(lambda x, y: (preprocess_input(x), y), num_parallel_calls=AUTOTUNE)

aug_p1 = tf.keras.Sequential([layers.RandomFlip("horizontal")])

# lighter augment for Phase 2 fine-tuning
aug_p2 = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.10),
    layers.RandomZoom(0.05),
])

train_ds_p1 = train_ds_pp.map(lambda x, y: (aug_p1(x, training=True), y),
                              num_parallel_calls=AUTOTUNE).prefetch(AUTOTUNE)
train_ds_p2 = train_ds_pp.map(lambda x, y: (aug_p2(x, training=True), y),
                              num_parallel_calls=AUTOTUNE).prefetch(AUTOTUNE)
val_ds = val_ds_pp.prefetch(AUTOTUNE)


base_model = DenseNet201(include_top=False, weights='imagenet', input_shape=(IMG_HEIGHT, IMG_WIDTH, 3))

inputs = Input(shape=(IMG_HEIGHT, IMG_WIDTH, 3))
x = base_model(inputs, training=False)
x = layers.GlobalAveragePooling2D()(x)

# optional BatchNorm head
x = layers.Dense(256, activation=None)(x)
x = layers.BatchNormalization()(x)
x = layers.Activation("relu")(x)
x = layers.Dropout(0.3)(x)
outputs = layers.Dense(num_classes, activation='softmax')(x)
model = Model(inputs, outputs)


#callbacks
best_model_file = "densenet201_fundus_best.h5"
cb = [
    callbacks.ModelCheckpoint(best_model_file,
                              monitor="val_accuracy",
                              save_best_only=True,
                              verbose=1),
    callbacks.ReduceLROnPlateau(monitor="val_accuracy",
                                patience=5,
                                factor=0.1,
                                min_lr=1e-7,
                                verbose=1),
    callbacks.EarlyStopping(monitor="val_accuracy",
                            patience=10,
                            verbose=1,
                            restore_best_weights=True)
]

# PHASE 1: HEAD-ONLY TRAINING
base_model.trainable = False
print("Trainable params (head only):")
model.summary()

model.compile(
    optimizer=optimizers.Adam(LR_P1),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

print("\n PHASE 1: training head only (no class-weights)")
history_p1 = model.fit(
    train_ds_p1,
    validation_data=val_ds,
    epochs=EPOCHS_P1,
    callbacks=cb
)

# PHASE 2: FINE-TUNE LAST LAYERS , Unfreeze last UNFREEZE_LAYERS of the base_model
for layer in base_model.layers[:-UNFREEZE_LAYERS]:
    layer.trainable = False
for layer in base_model.layers[-UNFREEZE_LAYERS:]:
    layer.trainable = True
    
print("\nTrainable layers in base_model after unfreeze:")
print(sum(l.trainable for l in base_model.layers), "of", len(base_model.layers))

# warming up the first few epochs without class-weights
model.compile(
    optimizer=optimizers.Adam(learning_rate=LR_P2),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)
print("\nPHASE 2(WARM-UP): fine-tuning (no class-weights)")
history_warm = model.fit(
    train_ds_p2,
    validation_data=val_ds,
    epochs=3,
    callbacks=cb
)

# with class-weights
model.compile(
    optimizer=optimizers.Adam(learning_rate=LR_P2),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)
print("\nPHASE 2: fine-tuning with class-weights")
history_p2 = model.fit(
    train_ds_p2,
    validation_data=val_ds,
    epochs=EPOCHS_P2-3,
    class_weight=class_weight_dict,
    callbacks=cb
)

val_accs = (
    history_p1.history.get('val_accuracy', []) +
    history_warm.history.get('val_accuracy', []) +
    history_p2.history.get('val_accuracy', [])
)
best_val_acc = max(val_accs) if val_accs else 0
print(f"\nHighest Validation Accuracy Achieved: {best_val_acc:.4f}")
