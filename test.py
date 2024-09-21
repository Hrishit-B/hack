from main import *
import zipfile

def linear_regression_testing(dataset_path, target_variable, output_path, performance):
    r = Regression(dataset_path, target_variable, output_path, performance)
    x = r.linear_regression()
    performance_fr["LinearRegression"] = x["LinearRegression"]

def lasso_regression_testing(dataset_path, target_variable, output_path, performance):
    r = Regression(dataset_path, target_variable, output_path, performance)
    x = r.lasso_regression()
    performance_fr["Lasso"] = x["Lasso"]

def decision_tree_regressor_testing(dataset_path, target_variable, output_path, performance):
    r = Regression(dataset_path, target_variable, output_path, performance)
    x = r.decision_tree_regressor()
    performance_fr["DecisionTreeRegressor"] = x["DecisionTreeRegressor"]

def random_forest_regressor_testing(dataset_path, target_variable, output_path, performance):
    r = Regression(dataset_path, target_variable, output_path, performance)
    x = r.random_forest_regressor()
    performance_fr["RandomForestRegressor"] = x["RandomForestRegressor"]

def gradient_boosting_regressor_testing(dataset_path, target_variable, output_path, performance):
    r = Regression(dataset_path, target_variable, output_path, performance)
    x = r.gradient_boosting_regressor()
    performance_fr["GradientBoostingRegressor"] = x["GradientBoostingRegressor"]

def zip_compile(files, zip_name):
        with zipfile.ZipFile(zip_name, 'w') as zipf:
            for file in files:
                zipf.write(file)
    
performance_fr = dict()

linear_regression_testing("winequality-red.csv", "quality", "linear.joblib", dict())
lasso_regression_testing("winequality-red.csv", "quality", "lasso.joblib", dict())
decision_tree_regressor_testing("winequality-red.csv", "quality", "decision_tree.joblib", dict())
random_forest_regressor_testing("winequality-red.csv", "quality", "random_forest.joblib", dict())
gradient_boosting_regressor_testing("winequality-red.csv", "quality", "gradient_boosting.joblib", dict())

print(performance_fr)
    
files = ["linear.joblib", "lasso.joblib"]
zip_file_name = "Regression.zip"
zip_compile(files, zip_file_name)