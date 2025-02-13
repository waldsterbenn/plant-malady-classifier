import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import os
import random


def predict_disease(model, image_path, class_names):
    """Make prediction on a single image"""
    img = tf.keras.preprocessing.image.load_img(
        image_path, target_size=(224, 224)
    )
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = img_array / 255.0  # Normalize the image
    img_array = tf.expand_dims(img_array, 0)

    predictions = model.predict(img_array)
    score = predictions[0]

    # Get top 3 predictions
    top_3_idx = np.argsort(score)[-3:][::-1]
    results = []
    for idx in top_3_idx:
        results.append({
            "class": class_names[idx],
            "confidence": float(score[idx])
        })

    return results

# Function to get a random test image


def get_random_test_image(data_dir):
    # Get all class directories
    class_dirs = [d for d in os.listdir(
        data_dir) if os.path.isdir(os.path.join(data_dir, d))]

    # Pick a random class
    random_class = random.choice(class_dirs)
    class_path = os.path.join(data_dir, random_class)

    # Get all images in that class
    images = [f for f in os.listdir(
        class_path) if f.casefold().endswith(('.jpg', '.jpeg', '.png'))]

    # Pick a random image
    random_image = random.choice(images)
    image_path = os.path.join(class_path, random_image)

    return image_path, random_class


def run_random_prediction():
    data_dir = "data_training/PlantVillage"
    # data_dir = "PlantVillage" # for Colab

    # Load the saved model
    model = tf.keras.models.load_model('plant_disease_model.h5')
    # Get class names
    class_names = sorted([item for item in os.listdir(data_dir)
                          if os.path.isdir(os.path.join(data_dir, item))])

    # Get a random test image
    test_image_path, true_class = get_random_test_image(data_dir)

    # Make prediction
    results = predict_disease(model, test_image_path, class_names)

    # Print results
    print(f"\nTest image: {test_image_path}")
    print(f"True class: {true_class}")
    print("\nTop 3 predictions:")
    for i, result in enumerate(results, 1):
        print(f"{i}. {result['class']}: {result['confidence']*100:.2f}%")

    # Display the image with top 3 predictions
    img = mpimg.imread(test_image_path)
    plt.figure(figsize=(8, 8))
    plt.imshow(img)
    plt.title(
        f"True: {true_class}\n1. {results[0]['class']}: {results[0]['confidence']*100:.2f}%\n2. {results[1]['class']}: {results[1]['confidence']*100:.2f}%\n3. {results[2]['class']}: {results[2]['confidence']*100:.2f}%")
    plt.axis('off')
    plt.show()


run_random_prediction()
