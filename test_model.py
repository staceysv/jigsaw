from tensorflow import keras
import numpy as np
import os
import wandb

MODEL_NAME = "trained_models"
# maybe we can still load from here?

run = wandb.init(project="jigsaw", PROJECT_NAME, job_type="inference")
model_at = run.use_artifact(MODEL_NAME + ":latest")
model_dir = model_at.download()
print("model: ", model_dir)
model = keras.models.load_model(model_dir)
model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
