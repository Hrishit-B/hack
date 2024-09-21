from main import *
import zipfile

def regression_testing():
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
    decision_tree_regressor_testing("winequality-red.csv", "quality", "decision_tree_r.joblib", dict())
    random_forest_regressor_testing("winequality-red.csv", "quality", "random_forest_r.joblib", dict())
    gradient_boosting_regressor_testing("winequality-red.csv", "quality", "gradient_boosting_r.joblib", dict())

    print(performance_fr)
        
    files = ["linear.joblib", "lasso.joblib", "decision_tree_r.joblib", "random_forest_r.joblib", "gradient_boosting_r.joblib"]
    zip_file_name = "Regression.zip"
    zip_compile(files, zip_file_name)

def classification_testing():
    def logistic_regression_testing(dataset_path, target_variable, output_path, performance):
        r = Classification(dataset_path, target_variable, output_path, performance)
        x = r.logistic_regression()
        performance_fr["LogisticRegression"] = x["LogisticRegression"]

    def naive_bayes_testing(dataset_path, target_variable, output_path, performance):
        r = Classification(dataset_path, target_variable, output_path, performance)
        x = r.naive_bayes_classification()
        performance_fr["NaiveBayes"] = x["NaiveBayes"]

    def gaussian_process_testing(dataset_path, target_variable, output_path, performance):
        r = Classification(dataset_path, target_variable, output_path, performance)
        x = r.gaussian_process_classification()
        performance_fr["GaussianProcess"] = x["GaussianProcess"]
    
    def support_vector_testing(dataset_path, target_variable, output_path, performance):
        r = Classification(dataset_path, target_variable, output_path, performance)
        x = r.support_vector_classification()
        performance_fr["SupportVector"] = x["SupportVector"]

    def decision_tree_classification_testing(dataset_path, target_variable, output_path, performance):
        r = Classification(dataset_path, target_variable, output_path, performance)
        x = r.decision_tree_classification()
        performance_fr["DecisionTreeClassification"] = x["DecisionTreeClassification"]

    def random_forest_classification_testing(dataset_path, target_variable, output_path, performance):
        r = Classification(dataset_path, target_variable, output_path, performance)
        x = r.random_forest_classification()
        performance_fr["RandomForestClassification"] = x["RandomForestClassification"]

    def gradient_boosting_Classification_testing(dataset_path, target_variable, output_path, performance):
        r = Classification(dataset_path, target_variable, output_path, performance)
        x = r.gradient_boosting_regressor()
        performance_fr["GradientBoostingClassification"] = x["GradientBoostingClassification"]

    def zip_compile(files, zip_name):
            with zipfile.ZipFile(zip_name, 'w') as zipf:
                for file in files:
                    zipf.write(file)
        
    performance_fr = dict()

    logistic_regression_testing("winequality-red.csv", "quality", "logistic.joblib", dict())
    naive_bayes_testing("winequality-red.csv", "quality", "naive_bayes.joblib", dict())
    gaussian_process_testing("winequality-red.csv", "quality", "gaussian_process.joblib", dict())
    support_vector_testing("winequality-red.csv", "quality", "support_vector.joblib", dict())
    decision_tree_classification_testing("winequality-red.csv", "quality", "decision_tree_c.joblib", dict())
    random_forest_classification_testing("winequality-red.csv", "quality", "random_forest_c.joblib", dict())
    gradient_boosting_Classification_testing("winequality-red.csv", "quality", "gradient_boosting_c.joblib", dict())

    print(performance_fr)
        
    files = ["logistic.joblib", "naive_bayes.joblib", "gaussian_process.joblib", "support_vector.joblib", "decision_tree.joblib", "randomforest.joblib", "gradientboosting.joblib"]
    zip_file_name = "Regression.zip"
    zip_compile(files, zip_file_name)
    
regression_testing()
classification_testing()