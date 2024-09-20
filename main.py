import joblib
from turtle import pd
from statistics import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import PolynomialFeatures

class regression:
    def __init__(self, dataset_path, target_variable, output_path):
        self.dataset_path = dataset_path
        self.target_variable = target_variable
        self.output_path = output_path
        
    def load_dataset(self):
        try:
            dataset = pd.read_csv(self.dataset_path)
            return dataset
        except FileNotFoundError:
            print(f"File {self.dataset_path} not found.")
            return None
    
    def regression_model_linear(self):
        dataset = self.load_dataset()
        if dataset is not None:
            X = dataset.drop(self.target_variable, axis=1)
            y = dataset[self.target_variable]
            
            # Split the dataset into training and testing sets
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            # Create a regression model
            model = LinearRegression()
            
            # Train the model
            model.fit(X_train, y_train)
            
            # Save the trained model to a file
            joblib.dump(model, self.model_path)
            
            # Predict the target values for the testing set
            y_pred = model.predict(X_test)
            
            # Evaluate the model's performance
            r2 = r2_score(y_test, y_pred)
            mse = mean_squared_error(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred)
            
            # Print the evaluation metrics
            print(f"R-squared: {r2:.2f}")
            print(f"Mean Squared Error: {mse:.2f}")
            print(f"Mean Absolute Error: {mae:.2f}")

    def regression_model_polynomial(self, degree=2):
        dataset = self.load_dataset()
        if dataset is not None:
            X = dataset.drop(self.target_variable, axis=1)
            y = dataset[self.target_variable]
        
            # Split the dataset into training and testing sets
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
            # Create polynomial features
            poly_features = PolynomialFeatures(degree=degree, include_bias=False)
            X_train_poly = poly_features.fit_transform(X_train)
            X_test_poly = poly_features.transform(X_test)
        
            # Create a polynomial regression model
            model = LinearRegression()
        
            # Train the model
            model.fit(X_train_poly, y_train)
        
            # Save the trained model and polynomial features to files
            joblib.dump(model, self.model_path)
            joblib.dump(poly_features, self.poly_features_path)
        
            # Predict the target values for the testing set
            y_pred = model.predict(X_test_poly)
        
            # Evaluate the model's performance
            r2 = r2_score(y_test, y_pred)
            mse = mean_squared_error(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred)
        
            # Print the evaluation metrics
            print(f"R-squared: {r2:.2f}")
            print(f"Mean Squared Error: {mse:.2f}")
            print(f"Mean Absolute Error: {mae:.2f}")