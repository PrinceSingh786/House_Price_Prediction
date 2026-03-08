import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import fetch_california_housing

data = fetch_california_housing()

df = pd.DataFrame(data.data, columns=data.feature_names)
df["Price"] = data.target
#df.head() - loads first five rows of datasets
#df.shape - tells number of rows and columns
#df.describe() - mean max min std deviation
# plt.figure(figsize=(10,9))
# sns.heatmap(df.corr(), annot=True)
# plt.show()
# sns.heatmap(df.corr(), annot=True)
# plt.show() #Without this line, the plot may not appear in some environments

#Day 5
X = df.drop("Price", axis=1) # drop price from input features
y = df["Price"]              # add target variable
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)    # X-features , y-target_values

from sklearn.linear_model import LinearRegression

model = LinearRegression()  # creating an instance of actual model

model.fit(X_train, y_train) # training model with training data
predictions = model.predict(X_test)  # making model to make prediction on testing data
print(predictions)
from sklearn.metrics import mean_squared_error

mse = mean_squared_error(y_test, predictions)

print("Mean Squared Error:", mse)
