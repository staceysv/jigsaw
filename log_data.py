import wandb

import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from wandb.keras import WandbCallback

wandb.init(project="jigsaw", name="log_data", job_type="data")

train_data = pd.read_csv("../../BigData/jigsaw/train.csv")
print(train_data.head())
small_set = train_data.sample(frac=0.1)
#small_set = train_data.iloc[:10000]

wandb.run.log({"sample_random_10%" : wandb.Table(dataframe=small_set)})


