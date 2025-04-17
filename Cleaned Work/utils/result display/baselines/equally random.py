import pandas as pd
import random

data = pd.read_csv("../data/train.csv")
data["model_classification"] = data.apply(lambda x : random.sample(["background","method","result"],1)[0],axis = 1)
data.to_csv("../results/Completely_random/output.csv", index=False)