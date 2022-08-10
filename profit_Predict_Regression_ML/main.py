import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
# creating dataset
dataset = pd.read_csv("startups.csv")

# label encoding for converting values to numbers
labelencoder = LabelEncoder()
dataset["State"] = labelencoder.fit_transform(dataset["State"].values)
# print(dataset.head())

# preparing data set for split
X = dataset.drop("Profit", axis=1)
y = dataset.loc[:, "Profit"]

# splitting train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

# linear regression
model = LinearRegression()

# training model
model.fit(X_train, y_train)

# making predictions
predictions = model.predict(X_test)

# visualizing results
comparison = pd.DataFrame({"Actual Profit": y_test, "Predicted Profit": predictions})

print("\n")
print(comparison.head())
print("\n")
print(comparison.tail())
