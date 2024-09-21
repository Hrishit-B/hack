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

def zip_compile(files, zip_name):
    with zipfile.ZipFile(zip_name, 'w') as zipf:
        for file in files:
            zipf.write(file)

print(linear_regression_testing("winequality-red.csv", "quality", "linear.joblib", dict()))
print(linear_regression_testing("winequality-red.csv", "quality", "lasso.joblib", dict()))
files = ["linear.joblib", "lasso.joblib"]
zip_file_name = "Regression.zip"
zip_compile(files, zip_file_name)