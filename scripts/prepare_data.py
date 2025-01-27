import pandas as pd
import zipfile
import os, shutil, pathlib

from sklearn.model_selection import train_test_split

# Seed for random functions
SEED = 42

# Function to retrieve class/label name from one-hot label data 
def extract_label_name(row):
    col_names = ['antelope_duiker', 'bird', 'blank', 'civet_genet', 'hog',
       'leopard', 'monkey_prosimian', 'rodent']
    for col in col_names:
        if row[col] == 1.0:
            return col
        
# Function to remove prefix from `filepath` URLs
def remove_prefix(row):
    return row['filepath'].split('/')[1]

# Function to create hierarchical data directory according to data subsets and labels
def create_data_directories(data_subsets, labels, original_dir, new_base_dir):
    for subset_name, subset in data_subsets.items():
        for label in labels:
            data_dir = new_base_dir /  subset_name / label
            os.makedirs(data_dir)
            fnames = list(subset[subset['label'] == label].apply(remove_prefix, axis=1))
            for fname in fnames:
                shutil.copyfile(src=original_dir / fname, dst=data_dir / fname)

# Zip file containing data
data_zip = 'species_data_original.zip'

# Extract data to root directory
with zipfile.ZipFile('species_data_original.zip', 'r') as zip_ref:
    zip_ref.extractall('.')

# File containing train features metadata
features_file = 'species_data_original/train_features.csv'
# File containing train labels
labels_file = 'species_data_original/train_labels.csv'

# Load into DataFrames
features = pd.read_csv(features_file)
labels = pd.read_csv(labels_file)

# Create new labels DataFrame containing named label instead of one-hot vector
labels = pd.concat([labels['id'], labels.apply(extract_label_name, axis=1)], axis=1)
labels.columns = ['id', 'label']

# Ensure feature and label instances are aligned
assert (features['id'] == labels['id']).all()
# Ensure instances are unique (for merge)
assert features['id'].nunique() == len(features)

# Join features and labels on id to create single DataFrame
data_df = pd.merge(features, labels, on='id')

# Data split proportions
train_p = 0.7
valid_p = 0.15
test_p = 0.15

# Split data into train, valid, and test sets
train_features_all, test_features, train_labels_all, test_labels = \
    train_test_split(
        features,
        labels,
        test_size=test_p,
        random_state=SEED,
        stratify=labels['label']
    )
train_features, valid_features, train_labels, valid_labels = \
    train_test_split(
        train_features_all,
        train_labels_all,
        test_size=(valid_p/(train_p + valid_p)),
        random_state=SEED,
        stratify=train_labels_all['label']
    )

# Reset indices
train_features = train_features.reset_index(drop=True)
train_labels = train_labels.reset_index(drop=True)
valid_features = valid_features.reset_index(drop=True)
valid_labels = valid_labels.reset_index(drop=True)
test_features = test_features.reset_index(drop=True)
test_labels = test_labels.reset_index(drop=True)

# Combine features metadata and label names into one DataFrame
train_df = pd.merge(train_features, train_labels, on='id')
valid_df = pd.merge(valid_features, valid_labels, on='id')
test_df = pd.merge(test_features, test_labels, on='id')

# Directory containing original data
original_dir = pathlib.Path('species_data_original/train_features')
# Destination where we want to set up hierarchical data directory
new_base_dir = pathlib.Path('species_data')
# Data split subsets
data_subsets = {
    'train': train_df,
    'valid': valid_df,
    'test': test_df
}
# Labels
labels = list(train_df['label'].unique())

# Create hierarchical data directory
create_data_directories(data_subsets, labels, original_dir, new_base_dir)