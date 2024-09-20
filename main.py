import joblib
from turtle import pd
from statistics import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.base import r2_score
from sklearn.metrics import mean_squared_error, mean_absolute_error

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
    
    def regression_model_linear(self.dataset_path):
        dataset = self.load_dataset()
        if dataset is not None:
            X = dataset.drop('target', axis=1)
            y = dataset['target']
            
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