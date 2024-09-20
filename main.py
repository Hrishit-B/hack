import joblib
from turtle import pd
from statistics import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import PolynomialFeatures

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
    
    def linear(self):
        dataset = self.load_dataset()
        if dataset is not None:
            X = dataset.drop(self.target_variable, axis=1)
            y = dataset[self.target_variable]

            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
            
            model = LinearRegression()
            model.fit(X_train, y_train)
            
            joblib.dump(model, self.model_path)
            
            y_pred = model.predict(X_test)
            
            r2 = r2_score(y_test, y_pred)
            mse = mean_squared_error(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred)
            
            print(f"R-squared: {r2:.2f}")
            print(f"Mean Squared Error: {mse:.2f}")
            print(f"Mean Absolute Error: {mae:.2f}")

    def polynomial(self, degree=2):
        dataset = self.load_dataset()
        if dataset is not None:
            X = dataset.drop(self.target_variable, axis=1)
            y = dataset[self.target_variable]

            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)

            X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
        
            poly_features = PolynomialFeatures(degree=degree, include_bias=False)
            X_train_poly = poly_features.fit_transform(X_train)
            X_test_poly = poly_features.transform(X_test)
        
            model = LinearRegression()     
            model.fit(X_train_poly, y_train)
        
            joblib.dump(model, self.model_path)
            joblib.dump(poly_features, self.poly_features_path)
        
            y_pred = model.predict(X_test_poly)
        
            r2 = r2_score(y_test, y_pred)
            mse = mean_squared_error(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred)
        
            print(f"R-squared: {r2:.2f}")
            print(f"Mean Squared Error: {mse:.2f}")
            print(f"Mean Absolute Error: {mae:.2f}")
            
    def regression_model_lasso(self, alpha=1.0):
        dataset = self.load_dataset()
        if dataset is not None:
            X = dataset.drop(self.target_variable, axis=1)
            y = dataset[self.target_variable]
        
            # Split the dataset into training and testing sets
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
            # Scale the features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
        
            # Create a Lasso regression model
            model = Lasso(alpha=alpha, random_state=42)
        
            # Train the model
            model.fit(X_train_scaled, y_train)
        
            # Save the trained model and scaler to files
            joblib.dump(model, self.model_path)
            joblib.dump(scaler, self.scaler_path)
        
            # Predict the target values for the testing set
            y_pred = model.predict(X_test_scaled)
        
            # Evaluate the model's performance
            r2 = r2_score(y_test, y_pred)
            mse = mean_squared_error(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred)
        
            # Print the evaluation metrics
            print(f"R-squared: {r2:.2f}")
            print(f"Mean Squared Error: {mse:.2f}")
            print(f"Mean Absolute Error: {mae:.2f}")
        
            # Print the number of features used by the model
            n_features_used = np.sum(model.coef_ != 0)
            print(f"Number of features used: {n_features_used}")