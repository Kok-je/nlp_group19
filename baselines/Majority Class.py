import pandas as pd
import random

data = pd.read_csv("../data/train.csv")
data["model_classification"] = data.apply(lambda x : "background",axis = 1)
data.to_csv("../results/Majority/output.csv", index=False)