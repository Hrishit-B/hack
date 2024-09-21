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

    def result_regression(dataset_path, target_variable):

        linear_regression_testing(dataset_path, target_variable, "linear.joblib", dict())
        lasso_regression_testing(dataset_path, target_variable, "lasso.joblib", dict())
        decision_tree_regressor_testing(dataset_path, target_variable, "decision_tree.joblib", dict())
        random_forest_regressor_testing(dataset_path, target_variable, "random_forest.joblib", dict())
        gradient_boosting_regressor_testing(dataset_path, target_variable, "gradient_boosting.joblib", dict())

        for k in performance_fr.keys():
            print(k)
            for v in performance_fr[k].keys():
                print("{}: {}".format(v, performance_fr[k][v]))
            print()

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

    def result_classification(dataset_path, target_variable):

        logistic_regression_testing(dataset_path, target_variable, "logistic.joblib", dict())
        naive_bayes_testing(dataset_path, target_variable, "naive_bayes.joblib", dict())
        gaussian_process_testing(dataset_path, target_variable, "gaussian_process.joblib", dict())
        support_vector_testing(dataset_path, target_variable, "support_vector.joblib", dict())
        decision_tree_classification_testing(dataset_path, target_variable, "decision_tree_c.joblib", dict())
        random_forest_classification_testing(dataset_path, target_variable, "random_forest_c.joblib", dict())
        gradient_boosting_Classification_testing(dataset_path, target_variable, "gradient_boosting_c.joblib", dict())

        for k in performance_fr.keys():
            print(k)
            for v in performance_fr[k].keys():
                print("{}: {}".format(v, performance_fr[k][v]))
            print()