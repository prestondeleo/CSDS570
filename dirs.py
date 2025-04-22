import os
ROOT_DIR = os.path.dirname(__file__)
DATASET_DIR = os.path.join(ROOT_DIR, 'data')
os.makedirs(DATASET_DIR, exist_ok=True)
GENERATED_DIR = os.path.join(ROOT_DIR, 'generated')
os.makedirs(GENERATED_DIR, exist_ok=True)
OUTPUT_DIR = os.path.join(ROOT_DIR, 'vae_output')
os.makedirs(OUTPUT_DIR, exist_ok=True)
MODEL_DIR = os.path.join(ROOT_DIR, 'model_weights')
os.makedirs(MODEL_DIR, exist_ok=True)

