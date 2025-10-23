import gradio as gr
import numpy as np
import tensorflow as tf
from tensorflow import keras
from PIL import Image

MODEL_PATH = "food_cls_efficientnet.keras"
CLASSES_PATH = "classes.txt"
IMG_SIZE = 224

INGRDIENTS_MAP ={
    "pho": ["rice noodles", "beef/chicken", "broth", "green onions", "star anise"],
    "com_tam": ["broken rice", "grilled pork/ribs", "fish sauce", "cucumber", "scallion oil"],
    "bun": ["vermicelli", "meat/spring rolls", "herbs", "peanuts", "fish sauce"]
}

print("Loading model...")
model = keras.models.load_model(MODEL_PATH)
print("Model loaded successfully!")

with open(CLASSES_PATH, 'r', encoding='utf-8') as f:
    class_names = [line.strip() for line in f.readlines()]
    
print(f"Classes: {class_names}")

def preprocess_image(image):
    """Preprocess input image"""
    
    if not isistance(image, Image.Image):
        image = Image.fromarray(image.astype('uint8'))
        
    image = image.resize((IMG_SIZE, IMG_SIZE))
    
    img_array = np.array(image)
    img_array = np.expand_dims(img_array, axis=0)
    
    img_array = tf.keras.applications.efficientnet.preprocess_input(img_array)
    return img_array

def predict(image):
    """Predict dish from image"""
    if image is None:
        return {
            "error": "Please provide an input image"
        }
    
    try: 
        processed_img = preprocess_image(image)
        
        predictions = model.predict(processed_img, verbose=0)
        predicted_index = np.argmax(predictions[0])
        confidence = float(predictions[0][predicted_idx])
        
        dish_name = class_names[predicted_idx]
        ingredients = INGRDIENTS_MAP.get(dish_name, [])
        
        result = {
            "dish_name": dish_name,
            "confidence": f"{confidence * 100:.2f}%",
            "typical_ingredients": ingredients,
            "all_predictions": {
                class_names[i]: f"{predictions[0][i] * 100:.2f}%" 
                for i in range(len(class_names))
            }
        }
        
        return result
    
    except Exception as e:
        return {
            "error": f"Processing error: {str(e)}"
        }
        
with gr.Blocks(title="Vietnamese Food Classification") as demo:
    gr.Markdown("""
    # Vietnamese Food Classifier
    
    Identify 3 popular Vietnamese dishes: **Pho**, **Com Tam** (Broken Rice), and **Bun** (Vermicelli) 
    
    Upload an image or capture from webcam to classify the dish and see typical ingredients
    """)
    
    with gr.Row():
        with gr.Column():
            with gr.Tab("Upload Image"):
                image_input = gr.Image(type="numpy", label="Select food image")
                upload_btn = gr.Button("Classify", variant="primary")
                
            with gr.Tab("Webcam"):
                webcam_input = gr.Image(source="webcam", type="numpy", label="Capture from webcam")
                webcam_btn = gr.Button("Classify", variant="primary")
        
        with gr.Column():
            output = gr.JSON(label="Classification Results")
            
    gr.Markdown("### Examples")
    gr.Markdown("Upload an image of Vietnamese food to test the model!")
    
    upload_btn.click(fn=predict, inputs=image_input, outputs=output)
    webcam_btn.click(fn=predict, inputs=webcam_input, outputs=output)

if __name__ == "__main__":
    demo.launch()