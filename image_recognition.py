import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.applications.efficientnet import preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt

# Charger le modèle pré-entraîné EfficientNetB0
model = EfficientNetB0(weights='imagenet')

# Fonction pour charger et prétraiter l'image
def load_and_preprocess_image(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    return img_array

# Fonction pour prédire la classe de l'image
def predict_image_class(img_path):
    img_array = load_and_preprocess_image(img_path)
    predictions = model.predict(img_array)
    decoded_predictions = decode_predictions(predictions, top=3)[0]
    return decoded_predictions

# Fonction pour afficher l'image et les prédictions
def display_image_with_predictions(img_path, predictions):
    img = image.load_img(img_path)
    plt.imshow(img)
    plt.axis('off')
    plt.title('Prédictions:')
    for i, (imagenet_id, label, score) in enumerate(predictions):
        plt.title(f'Prédictions:\n{i + 1}: {label} (Score: {score:.2f})', fontsize=10)
    plt.show()

# Main function to test the image recognition
def main(image_path):
    # Get predictions
    predictions = predict_image_class(image_path)

    # Display the image with predictions
    display_image_with_predictions(image_path, predictions)

# Path to an image file (Make sure to replace this with your own image path)
main('images/french-pug.jpg')  # Change this to your image file path
main('images/license-plate.jpg')
main('images/garden-gnome-Eiffel-tower.png')
main('images/garden-gnome.webp')