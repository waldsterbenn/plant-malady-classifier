import os
import matplotlib.pyplot as plt
from tensorflow.keras import layers, models, Input, saving
import tensorflow as tf

from infrence import run_random_prediction


data_dir = "data_training/PlantVillage"
# data_dir = "/kaggle/input"


def get_class_names(train_data_dir):
    """Get class names from directory structure"""
    class_names = sorted([item for item in os.listdir(train_data_dir)
                          if os.path.isdir(os.path.join(train_data_dir, item))])

    if not class_names:
        raise ValueError(
            "No valid class directories found in the data directory")
    print(f"Found {len(class_names)} classes: {class_names}")

    return class_names


def create_model(num_classes):
    """Create a CNN model for plant disease classification"""
    inputs = Input(shape=(224, 224, 3))

    # Base convolutional layers with batch normalization
    x = layers.Conv2D(64, 3, padding='same')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.MaxPooling2D()(x)

    x = layers.Conv2D(128, 3, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.MaxPooling2D()(x)

    x = layers.Conv2D(256, 3, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.MaxPooling2D()(x)

    x = layers.Conv2D(512, 3, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.MaxPooling2D()(x)

    # Dense layers
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(512, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)

    model = models.Model(inputs=inputs, outputs=outputs)
    return model


def prepare_data(data_dir, img_height=224, img_width=224, batch_size=32):
    """Prepare and augment the dataset"""
    # Get class names first
    class_names = get_class_names(data_dir)

    # Data augmentation
    data_augmentation = tf.keras.Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.2),
        layers.RandomZoom(0.1),
        layers.RandomBrightness(0.2),
        layers.RandomContrast(0.2),
    ])

    # Create datasets
    train_ds = tf.keras.preprocessing.image_dataset_from_directory(
        data_dir,
        validation_split=0.2,
        subset="training",
        seed=123,
        image_size=(img_height, img_width),
        batch_size=batch_size,
        label_mode='categorical',
        shuffle=True
    )

    val_ds = tf.keras.preprocessing.image_dataset_from_directory(
        data_dir,
        validation_split=0.2,
        subset="validation",
        seed=123,
        image_size=(img_height, img_width),
        batch_size=batch_size,
        label_mode='categorical',
        shuffle=True
    )

    # Normalize the data and apply augmentation
    def preprocess_data(images, labels, training=True):
        # Normalize images
        images = tf.cast(images, tf.float32) / 255.0
        if training:
            images = data_augmentation(images)
        return images, labels

    train_ds = train_ds.map(
        lambda x, y: preprocess_data(x, y, training=True),
        num_parallel_calls=tf.data.AUTOTUNE
    )
    val_ds = val_ds.map(
        lambda x, y: preprocess_data(x, y, training=False),
        num_parallel_calls=tf.data.AUTOTUNE
    )

    # Configure dataset for performance
    train_ds = train_ds.cache().prefetch(buffer_size=tf.data.AUTOTUNE)
    val_ds = val_ds.cache().prefetch(buffer_size=tf.data.AUTOTUNE)

    return train_ds, val_ds, class_names


def train_model(model, train_ds, val_ds, epochs=20):
    """Train the model with data augmentation"""
    # Compile model with appropriate learning rate
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    model.compile(
        optimizer=optimizer,
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    # Create callbacks
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True
    )

    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.2,
        patience=5,
        min_lr=1e-6
    )

    # Train the model
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        callbacks=[early_stopping, reduce_lr]
    )

    return history


def plot_training_results(history):
    """Plot training and validation accuracy/loss"""
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs_range = range(len(acc))

    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.show()


# Prepare data
train_ds, val_ds, class_names = prepare_data(data_dir, batch_size=32)

# Create and train model
model = create_model(num_classes=len(class_names))
history = train_model(model, train_ds, val_ds, epochs=2)

# Save the model
model.save('plant_disease_model.h5')

# Visualize results
plot_training_results(history)

# Test the model with an example image from validation set
run_random_prediction()
