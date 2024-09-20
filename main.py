import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error, mean_squared_log_error, r2_score
import joblib

class Regression:
    def __init__(self, dataset_path, target_variable, output_path):
        self.dataset_path = dataset_path
        self.target_variable = target_variable
        self.output_path = output_path
        
    def load_dataset(self):
        try:
            try:
                dataset = pd.read_csv(self.dataset_path)
            except:
                pass

            try:
                dataset = pd.read_excel(self.dataset_path)
            except:
                pass

            return dataset
        
        except FileNotFoundError:
            print("File {} not found".format(self.dataset_path))
            return None
        
        except:
            print("Some error has occured")
            return None
    
    def preprocessing(self):
        dataset = self.load_dataset()
        if dataset is not None:
            X = dataset.drop(self.target_variable, axis=1)
            y = dataset[self.target_variable]

            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
        
            X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
            return X_train, X_test, y_train, y_test
        else:
            return
        
    def save_model(self, model):
        joblib.dump(model, self.model_path)

    def performance(self, y_test, y_pred):
        r2 = r2_score(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, y_pred)
        mape = mean_absolute_percentage_error(y_test, y_pred)
        msle = mean_squared_log_error(y_test, y_pred)
        rmsle = np.sqrt(msle)

    def linear_regression(self):
        X_train, X_test, y_train, y_test = self.preprocessing(self)
            
        model = LinearRegression()
        model.fit(X_train, y_train)
            
        y_pred = model.predict(X_test)
        self.performance(self, y_test, y_pred)

        self.save_model(self, model)

    def polynomial_regression(self):
        X_train, X_test, y_train, y_test = self.preprocessing(self)
        
        model = PolynomialFeatures(degree=3)
        X_poly = model.fit_transform(X_train)
        model.fit(X_poly, y_train)

        y_pred = model.predict(X_test)
        self.performance(self, y_test, y_pred)

        self.save_model(self, model)
            
    def lasso_regression(self):
        X_train, X_test, y_train, y_test = self.preprocessing(self)
        
        model = Lasso(selection='random', random_state=42)
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_test)
        self.performance(self, y_test, y_pred)

        self.save_model(self, model)

    def decision_tree_regressor(self):
        X_train, X_test, y_train, y_test = self.preprocessing(self)
            
        model = DecisionTreeRegressor(criterion="friedman_mse", random_state=42)
        model.fit(X_train, y_train)
            
        y_pred = model.predict(X_test)
        self.performance(self, y_test, y_pred)

        self.save_model(self, model)

    def random_forest_regressor(self):
        X_train, X_test, y_train, y_test = self.preprocessing(self)
            
        model = RandomForestRegressor(criterion="friedman_mse", random_state=42)
        model.fit(X_train, y_train)
            
        y_pred = model.predict(X_test)
        self.performance(self, y_test, y_pred)

        self.save_model(self, model)

    def gradient_boosting_regressor(self):
        X_train, X_test, y_train, y_test = self.preprocessing(self)
            
        model = GradientBoostingRegressor(loss="huber", criterion="friedman_mse", random_state=42)
        model.fit(X_train, y_train)
            
        y_pred = model.predict(X_test)
        self.performance(self, y_test, y_pred)

        self.save_model(self, model)

