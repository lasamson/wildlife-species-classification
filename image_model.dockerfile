FROM tensorflow/serving:2.18.0

COPY bin/saved_model /models/species_model/1
ENV MODEL_NAME="species_model"