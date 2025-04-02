import pandas as pd
import random

data = pd.read_csv("../data/train.csv")
data["model_classification"] = data.apply(lambda x : random.choices(["background","method","result"],weights = [4840,2266,1088], k=1)[0]
                                          ,axis = 1)
data.to_csv("../results/Proportionally_random/output.csv", index=False)