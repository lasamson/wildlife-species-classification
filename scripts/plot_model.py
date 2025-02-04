import pathlib
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import plot_model

# Function to create model plot and save as PNG file
def save_model_plot(model_path, model_name):
    model = load_model(model_path)
    plot_model(
        model,
        to_file=f'img/{model_name}.png',
        show_shapes=True,
        show_dtype=False,
        show_layer_names=False,
        rankdir="TB",
        expand_nested=False,
        dpi=200,
        show_layer_activations=True,
        show_trainable=False
    )

# Path to the final model
model_dir = pathlib.Path('bin/custom/')
model_file = 'custom_dropout_0.5_100_0.846_0.521.keras'
model_path = model_dir / model_file

# Plot the final model
save_model_plot(model_path, 'small_cnn')