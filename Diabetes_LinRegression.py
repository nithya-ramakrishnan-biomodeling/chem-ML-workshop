import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler

#X, y = load_diabetes(return_X_y=True)
diabetes_df = load_diabetes(as_frame=True)
print(diabetes_df.frame.head())
print(diabetes_df.frame.columns)
X = diabetes_df.data[["age","bmi","bp","s4","s5","s6","s1","s2","s3"]].values
y=diabetes_df["target"]
#model = RandomForestRegressor()
#model=LinearRegression()
model=GradientBoostingRegressor()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=999)
print(X_train.shape)

#scaler = StandardScaler()
scaler=MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

model.fit(X_train_scaled,y_train)
y_test_pred=model.predict(X_test_scaled)
print(r2_score(y_test,y_test_pred))

kf = KFold(n_splits=10, shuffle=True, random_state=42)

scores = cross_val_score(model, X, y, cv=kf, scoring='r2')

print(scores)
print("Mean R2 across 5 folds ", scores.mean())
print("Std of R2 across 5 folds ", scores.std())
