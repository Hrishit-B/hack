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
        performance_fr["GaussianNB"] = x["GaussianNB"]
    
    def support_vector_testing(dataset_path, target_variable, output_path, performance):
        r = Classification(dataset_path, target_variable, output_path, performance)
        x = r.support_vector_classification()
        performance_fr["SVC"] = x["SVC"]

    def decision_tree_classification_testing(dataset_path, target_variable, output_path, performance):
        r = Classification(dataset_path, target_variable, output_path, performance)
        x = r.decision_tree_classification()
        performance_fr["DecisionTreeClassifier"] = x["DecisionTreeClassifier"]

    def random_forest_classification_testing(dataset_path, target_variable, output_path, performance):
        r = Classification(dataset_path, target_variable, output_path, performance)
        x = r.random_forest_classification()
        performance_fr["RandomForestClassifier"] = x["RandomForestClassifier"]

    def gradient_boosting_classification_testing(dataset_path, target_variable, output_path, performance):
        r = Classification(dataset_path, target_variable, output_path, performance)
        x = r.gradient_boosting_classification()
        performance_fr["GradientBoostingClassifier"] = x["GradientBoostingClassifier"]

    def zip_compile(files, zip_name):
            with zipfile.ZipFile(zip_name, 'w') as zipf:
                for file in files:
                    zipf.write(file)

    performance_fr = dict()

    def result_classification(input_path, target):

        logistic_regression_testing(input_path, target, "logistic.joblib", dict())
        naive_bayes_testing(input_path, target, "naive_bayes.joblib", dict())
        support_vector_testing(input_path, target, "support_vector.joblib", dict())
        decision_tree_classification_testing(input_path, target, "decision_tree_c.joblib", dict())
        random_forest_classification_testing(input_path, target, "random_forest_c.joblib", dict())
        gradient_boosting_classification_testing(input_path, target, "gradient_boosting_c.joblib", dict())

        for k in performance_fr.keys():
            print(k)
            for v in performance_fr[k].keys():
                print("{}: {}".format(v, performance_fr[k][v]))
            print()