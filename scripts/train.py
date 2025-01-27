import pathlib
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.utils import image_dataset_from_directory

# Function to load train, validation, and test datasets from directory
def load_datasets(batch_size=32):
    new_base_dir = pathlib.Path('species_data')
    
    train_dataset = image_dataset_from_directory(
        new_base_dir / 'train',
        image_size=(150, 150),
        batch_size=batch_size
    )
    validation_dataset = image_dataset_from_directory(
        new_base_dir / 'valid',
        image_size=(150, 150),
        batch_size=batch_size
    )
    test_dataset = image_dataset_from_directory(
        new_base_dir / 'test',
        image_size=(150, 150),
        batch_size=batch_size
    )

    return train_dataset, validation_dataset, test_dataset

# Function to generate instance of model architecture
def make_dropout_model(drop_rate=0.2):
    inputs = keras.Input(shape=(150, 150, 3))
    x = layers.Rescaling(1./255)(inputs)
    x = layers.Conv2D(filters=32, kernel_size=3, activation='relu')(x)
    x = layers.MaxPooling2D(pool_size=2)(x)
    x = layers.Conv2D(filters=64, kernel_size=3, activation='relu')(x)
    x = layers.MaxPooling2D(pool_size=2)(x)
    x = layers.Conv2D(filters=64, kernel_size=3, activation='relu')(x)
    x = layers.MaxPooling2D(pool_size=2)(x)
    x = layers.Conv2D(filters=128, kernel_size=3, activation='relu')(x)
    x = layers.MaxPooling2D(pool_size=2)(x)
    x = layers.Conv2D(filters=256, kernel_size=3, activation='relu')(x)
    x = layers.Flatten()(x)
    x = layers.Dropout(drop_rate)(x)
    x = layers.Dense(64, activation='relu')(x)
    x = layers.Dropout(drop_rate)(x)
    x = layers.Dense(32, activation='relu')(x)
    x = layers.Dropout(drop_rate)(x)
    outputs = layers.Dense(8, activation='softmax')(x)
    
    model = keras.Model(inputs=inputs, outputs=outputs)
    return model

# Directory to save models
model_dir = pathlib.Path('bin/custom/')
# Best model file name
model_file = 'best_model.keras'
# Best model path
model_path = model_dir / model_file

# Load datasets for each split
train_dataset, validation_dataset, test_dataset = load_datasets(batch_size=32)
# Generate model instance
model = make_dropout_model(drop_rate=0.5)
# Compile the model
model.compile(
    loss='sparse_categorical_crossentropy',
    optimizer=keras.optimizers.Adam(learning_rate=1e-4),
    metrics=['accuracy']
)

# Model checkpoint callback to save best model
callbacks = [
    keras.callbacks.ModelCheckpoint(
        filepath=model_path,
        save_best_only=True,
        monitor='val_loss',
        mode='min'
    )
]

# Fit the model
model.fit(
    train_dataset,
    epochs=150,
    validation_data=validation_dataset,
    callbacks=callbacks
)

# Load best model and evaluate on test dataset
final_model = keras.models.load_model(model_path)
test_loss, test_acc = final_model.evaluate(test_dataset)

# Output test set accuracy and loss
print(f'Test accuracy: {test_acc:.3f}')
print(f'Test loss: {test_loss:.3f}')