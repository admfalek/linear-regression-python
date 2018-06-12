# Data Preprocessing Template
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression

# importing the dataset
dataset = pd.read_csv('Salary_Data.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 1].values

# splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 1/3, random_state = 0)

# fitting simple linear regression to the training set
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# predicting test set results
y_pred = regressor.predict(X_test)

# training data visualization
# real data
plt.scatter(X_train, y_train, color='red')
# linear regression curve - predicted values
plt.plot(X_train, regressor.predict(X_train), color='green')
plt.title('Salary vs Experience (Training set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()

# test data visualization
# real data
plt.scatter(X_test, y_test, color='red')
# linear regression curve - predicted values
plt.plot(X_train, regressor.predict(X_train), color='green')
plt.title('Salary vs Experience (Test set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()

# Feature Scaling
"""from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
sc_y = StandardScaler()
y_train = sc_y.fit_transform(y_train)"""