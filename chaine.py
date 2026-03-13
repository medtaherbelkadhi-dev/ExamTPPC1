from sklearn.linear_model import LinearRegression
import numpy as np

X = np.array([[10,20],[15,30],[20,40],[25,50]])
y = np.array([100,80,60,50])

model = LinearRegression().fit(X, y)
print(model.predict([[18,35]]))

