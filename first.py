import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns     
from sklearn.datasets import fetch_california_housing

data = fetch_california_housing()

df = pd.DataFrame(data.data, columns=data.feature_names)
df["Price"] = data.target

df.head()