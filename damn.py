from main import *

def linear_regression_testing(dataset_path, target_variable, output_path, performance):
    r = Regression(dataset_path, target_variable, output_path, performance)
    r.linear_regression()
    x = r.lasso_regression()
    return x

print(linear_regression_testing("winequality-red.csv", "quality", "linear.joblib", dict()))
