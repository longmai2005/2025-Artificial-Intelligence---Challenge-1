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
    print("\n=== Evaluating on test set ===")

    y_true = []
    y_pred = []
    
    for images, labels in test_ds:
        predictions = model.predict(images, verbose=0)
        y_true.extend(np.argmax(labels.numpy(), axis=1))
        y_pred.extend(np.argmax(predictions, axis=1))
        
    print("\nClassification Report:")
    report = classification_report(y_true, y_pred, target_names=class_names)
    print(report)
    
    cm = confusion_matrix(y_true, y_pred)
    
    with open("confusion_matrix.json", "w") as f:
        f.write(','.join(class_names) + '\n')
        for i, row in enumerate(cm):
            f.write(class_names[i] + ',' + ','.join(map(str, row)) + '\n')
            
    print("\nConfusion matrix saved to: confusion_matrix.csv")
    
    test_loss, test_acc = model.evaluate(test_ds, verbose=0)
    
    metrics = {
        'test_accuracy': float(test_acc),
        'test_loss': float(test_loss),
        'num_classes': len(class_names),
        'classes': class_names 
    }
    
    with open ('metrics.json', 'w', encoding ='utf-8') as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
        
    print(f"✓ Metrics saved tro: metrics.json")
    print(f"Test Accuracy: {test_acc:.4f}")
    print(f"Test Loss: {test_loss:.4f}")
    
def main():
    print("=" * 60)
    print("FOO CLASSIFICATION TRAINING -3 classes")
    print("=" * 60)
    
    train_dir = os.path.join(DATA_DIR, "train")
    test_dir = os.path.join(DATA_DIR, "test")  
    
    if not os.path.exists(train_dir):
        print(f"ERROR: Directory {train_dir} not found!")
        print("Pleaase run bootstrap.sh first and add your images.")
        
        
    print(f"\nLoading datasets from {DATA_DIR}...")
    train_ds_raw = create_dataset(train_dir, 'training', validation_split=0.15)
    val_ds = create_dataset(train_dir, 'validation', validation_split=0.15)
    
    class_names = sorted(train_ds_raw.class_names)
    num_classes = len(class_names)
    
    print(f"\nClasses found: {class_names}")
    print(f"Number of classes: {num_classes}")
    
    with open(CLASSES_PATH, 'w', encoding='utf-8') as f:
        f.write('\n'.join(class_names))
    print(f"✓ Class names saved to: {CLASSES_PATH}")
    
    train_ds = prepare_dataset(train_ds_raw, augement=True)
    val_ds = prepare_dataset(val_ds, augement=False)
    
    test_ds = tf.keras.preprocessing.image_dataset_from_directory(
        test_dir,
        image_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        label_mode='categorical'
    )
    test_ds = prepare_dataset(test_ds, augement=False)
    
    print("\nBuilding EfficientNetB0 model...")
    model, base_model = build_model(num_classes)
    print(f"Model built: {model.count_params():,} parameters")
    
    history1 = train_phase1(model, train_ds, val_ds)
    history2 = train_phase2(model, base_model, train_ds, val_ds)    
    
    print(f"\nSaving trained model to: {MODEL_PATH} ...")
    model.save(MODEL_PATH)
    print("✓ Model saved to: {MODEL_PATH}")
    
    evaluate_model(model, test_ds, class_names)
    
    print("\n" + "=" * 60)
    print("TRAINING COMPLETED SUCCESSFULLY!")
    print("=" * 60) 
    print(f"\nGeneratd files:")
    print(f"  • {MODEL_PATH}")
    print(f"  • {CLASSES_PATH}")
    print(f"  • confusion_matrix.csv")
    print(f"  • metrics.json")
    print(f"\nNext steps:")
    print(f"  1. Run: python plot_cm.py (to visualize confusion matrix)")
    print(f"  2. Run: python app.py (to launch Gradio demo)")\
        
if __name__ == "__main__":
    main()