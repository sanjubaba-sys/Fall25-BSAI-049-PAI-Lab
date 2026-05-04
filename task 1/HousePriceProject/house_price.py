import pandas as pd
from sklearn.linear_model import LinearRegression

train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

X = train[["LotArea","OverallQual","YearBuilt"]]
y = train["SalePrice"]

model = LinearRegression()
model.fit(X,y)

X_test = test[["LotArea","OverallQual","YearBuilt"]]

prediction = model.predict(X_test)

result = pd.DataFrame({
    "Id": test["Id"],
    "SalePrice": prediction
})

result.to_csv("submission.csv",index=False)

print("submission file created")
