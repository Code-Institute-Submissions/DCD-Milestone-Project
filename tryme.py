# ### SIMPLE ###

# import numpy as np
# import matplotlib.pyplot as plt
# import pandas as pd

# dataset = pd.read_csv('salary.csv')
# X = dataset.iloc[:,:-1].values
# y = dataset.iloc[:, 1].values

# from sklearn.cross_validation import train_test_split
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 1/3, random_state = 0) 

# from sklearn.linear_model import LinearRegression
# regressor = LinearRegression()         
# regressor.fit(X_train,y_train)

# y_pred = regressor.predict(X_test)

# plt.scatter(X_train, y_train, color = 'red')
# plt.plot(X_train, regressor.predict(X_train), color = 'blue')
# plt.title('Salary vs Experience(Training set)')      
# plt.xlabel('Years of experience')
# plt.ylabel('Salary')            
# plt.show()

# plt.scatter(X_test, y_test, color = 'red')
# plt.plot(X_train, regressor.predict(X_train), color = 'blue')
# plt.title('Salary vs Experience(Test set)')      
# plt.xlabel('Years of experience')
# plt.ylabel('Salary')            
# plt.show()

### Multiple ### 

# import numpy as np
# import matplotlib.pyplot as plt
# import pandas as pd

# dataset = pd.read_csv('50_Startups.csv')
# X = dataset.iloc[:, :-1].values
# y = dataset.iloc[:, 4].values

# from sklearn.preprocessing import LabelEncoder, OneHotEncoder
# labelencoder_X = LabelEncoder()
# X[:, 3] = labelencoder_X.fit_transform(X[:, 3])    
# onehotencoder = OneHotEncoder(categorical_features = [3])
# X = onehotencoder.fit_transform(X).toarray()

# X = X[:,1:]

# from sklearn.cross_validation import train_test_split
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# import statsmodels.formula.api as sm

# X = np.append(arr = np.ones((50,1)).astype(int), values = X, axis = 1)

# X_opt = X[:,[0,1,2,3,4,5]] 
# regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
# regressor_OLS.summary()


###  Polynomial ###

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:,1:2].values
y = dataset.iloc[:, 2].values

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

poly_reg = PolynomialFeatures(degree = 4)          
X_poly = poly_reg.fit_transform(X)                     
lin_reg = LinearRegression()
lin_reg.fit(X_poly, y)

# X_grid = np.arange(min(X), max(X), 0.1)   
# X_grid = X_grid.reshape(len(X_grid),1)         
# plt.scatter(X,y, color = 'red')
# plt.plot(X_grid, lin_reg.predict(poly_reg.fit_transform(X_grid)), color = 'blue')    
# plt.title('Reality Check (Polynomial Regression)')
# plt.xlabel('Position Level')
# plt.ylabel('Salary')
# plt.show() 

lin_reg.predict(poly_reg.fit_transform(6.5))

# ### Support Vector Machine

# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt

# dataset = pd.read_csv("Position_Salaries.csv")
# X = dataset.iloc[:,1:2].values
# y = dataset.iloc[:,2].values

# from sklearn.preprocessing import StandardScaler
# sc_X = StandardScaler()
# X = sc_X.fit_transform(X)
# sc_y = StandardScaler()
# y = sc_y.fit_transform(y)

# from sklearn.svm import SVR
# regressor = SVR(kernel = 'rbf')
# regressor.fit(X,y)

# plt.scatter(X,y,color = 'red')
# plt.plot(X, regressor.predict(X), color = 'blue')
# plt.show()

# y_pred = sc_y.inverse_transform(regressor.predict(sc_X.transform(np.array([[6.5]]))))

# ### Random Forest Regression

# import numpy as np
# import matplotlib.pyplot as plt
# import pandas as pd

# dataset = pd.read_csv('Position_Salaries.csv')
# X = dataset.iloc[:, 1:2].values
# y = dataset.iloc[:, 2].values

# from sklearn.ensemble import RandomForestRegressor
# regressor = RandomForestRegressor(n_estimators = 600, random_state = 0)
# regressor.fit(X,y)

# y_pred = regressor.predict(6.5)

# X_grid = np.arange(min(X), max(X), 0.01)
# X_grid = X_grid.reshape((len(X_grid), 1))
# plt.scatter(X, y, color = 'red')
# plt.plot(X_grid, regressor.predict(X_grid), color = 'blue')
# plt.title('Reality Check (Random Forest Regression Model)')
# plt.xlabel('Position level')
# plt.ylabel('Salary')
# plt.show()

# ### IMPUTER

# from sklearn.preprocessing import Imputer
# imputer = Imputer(missing_values = "NaN", strategy = "mean", axis = 0)
# imputer = imputer.fit(X[:,1:3])
# X[:,1:3] = imputer.transform(X[:,1:3])































































