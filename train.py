import os
import json
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.metrics import classification_report, confusion_matrix
import random

SEED = 42
random.seed(SEED)
tf.random.set_seed(SEED)
np.random.seed(SEED)

IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS_PHASE1 = 8
EPOCHS_PHASE2 = 5
DATA_DIR = "data"
MODEL_PATH = "food_cls_efficientnet.keras"
CLASSES_PATH = "classes.txt"

def create_dataset(directory, subset, validation_split=0.15, seed=SEED):
    return tf.keras.preprocessing.image_dataset_from_directory(
        directory,
        validation_split=validation_split,
        subset=subset,
        seed=seed,
        image_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE
        label_mode='categorical'
    )

def augment_layer():
    return keras.Sequential(
        [
            layers.RandomFlip("horizontal"),
            layers.RandomRotation(0.05),
            layers.RandomZoom(0.1),
        ]
    )

def prepare_dataset(ds, augement=False, cache=True):
    if cache: 
        ds = ds.cache()
    if augement:
        aug = augment_layer()
        ds = ds.map(lambda x, y: (aug(x, training=True), y), 
                        num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.prefetch(buffer_size=tf.data.AUTOTUNE)
    return ds

def build_model(num_classes, input_shape=(IMG_SIZE, IMG_SIZE, 3)):
    base_model = tf.keras.applications.EfficientNetB0(
        include_top=False,
        weights="imagenet",
        input_shape=input_shape,
        pooling='avg'
    )
    base_model.trainable = False

    inputs = keras.Input(shape=input_shape)
    x = tf.keras.applications.efficientnet.preprocess_input(inputs)
    x = base_model(x, training=False)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    model = keras.Model(inputs, outputs)
    
    return model, base_model 

def train_phase1(model, train_ds, val_ds):
    print("\n=== PHASE 1: Training with frozen base ===")
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS_PHASE1,
        verbose=1
    )
    return history

def train_phase2(model, base_model, train_ds, val_ds):
    print("\n=== PHASE 2: Fine-tune last 20 layers ===")
    base_model.trainable = True
    
    for layer in base_model.layers[:-20]:
        layer.trainable = False
        
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-5),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS_PHASE2,
        verbose=1
    )
    return history

def evaluate_model(model, test_ds, class_names):
    print(})
        
        