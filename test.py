from main import *
import zipfile

def linear_regression_testing(dataset_path, target_variable, output_path, performance):
    r = Regression(dataset_path, target_variable, output_path, performance)
    x = r.linear_regression()
    return x

def lasso_regression_testing(dataset_path, target_variable, output_path, performance):
    r = Regression(dataset_path, target_variable, output_path, performance)
    x = r.lasso_regression()
    return x

def polynomial_regression_testing(dataset_path, target_variable, output_path, performance):
    r = Regression(dataset_path, target_variable, output_path, performance)
    x = r.polynomial_regression()
    return x

def decision_tree_regressor_testing(dataset_path, target_variable, output_path, performance):
    r = Regression(dataset_path, target_variable, output_path, performance)
    x = r.decision_tree_regressor()
    return x

def random_forest_regressor_testing(dataset_path, target_variable, output_path, performance):
    r = Regression(dataset_path, target_variable, output_path, performance)
    x = r.random_forest_regressor()
    return x

def gradient_boosting_regressor_testing(dataset_path, target_variable, output_path, performance):
    r = Regression(dataset_path, target_variable, output_path, performance)
    x = r.gradient_boosting_regressor()
    return x

def zip_compile(files, zip_name):
        with zipfile.ZipFile(zip_name, 'w') as zipf:
            for file in files:
                zipf.write(file)
    
print(linear_regression_testing("winequality-red.csv", "quality", "linear.joblib", dict()))
# print(polynomial_regression_testing("winequality-red.csv", "quality", "polynomial.joblib", dict()))
print(lasso_regression_testing("winequality-red.csv", "quality", "lasso.joblib", dict()))
print(decision_tree_regressor_testing("winequality-red.csv", "quality", "decision_tree.joblib", dict()))
print(random_forest_regressor_testing("winequality-red.csv", "quality", "random_forest.joblib", dict()))
print(gradient_boosting_regressor_testing("winequality-red.csv", "quality", "gradient_boosting.joblib", dict()))
    
files = ["linear.joblib", "lasso.joblib"]
zip_file_name = "Regression.zip"
zip_compile(files, zip_file_name)