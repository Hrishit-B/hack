import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score

from sklearn.linear_model import LinearRegression, Lasso
from sklearn.preprocessing import PolynomialFeatures
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error, mean_squared_log_error, r2_score

from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, balanced_accuracy_score, precision_score, average_precision_score, recall_score, jaccard_score, f1_score, roc_auc_score

from sklearn.cluster import KMeans, DBSCAN,AgglomerativeClustering
from sklearn_extra.cluster import KMedoids
from sklearn.mixture import GaussianMixture
from sklearn.metrics.cluster import contingency_matrix
from sklearn.metrics import adjusted_mutual_info_score, adjusted_rand_score, calinski_harabasz_score, completeness_score, davies_bouldin_score, fowlkes_mallows_score, homogeneity_score, mutual_info_score, normalized_mutual_info_score, rand_score, silhouette_score, v_measure_score

from sklearn.feature_selection import VarianceThreshold, SelectKBest, chi2, mutual_info_classif, f_classif, RFE, ExhaustiveFeatureSelector

class Regression:
    def __init__(self, dataset_path, target_variable, output_path):
        self.dataset_path = dataset_path
        self.target_variable = target_variable
        self.output_path = output_path

    def load_dataset(self):
        try:
            dataset = pd.read_csv(self.dataset_path)
            return dataset
        
        except FileNotFoundError:
            print("File {} not found".format(self.dataset_path))
            return None
        
        except:
            print("Some error has occured")
            return None
    
    def preprocessing(self):
        dataset = self.load_dataset()
        if dataset is not None:
            X = dataset.drop(self.target_variable, axis=1)
            y = dataset[self.target_variable]

            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
        
            X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
            return X_train, X_test, y_train, y_test
        else:
            return
    
    def save_model(self, model):
        joblib.dump(model, self.model_path)

    def performance(self, y_test, y_pred):
        r2 = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        mape = mean_absolute_percentage_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        msle = mean_squared_log_error(y_test, y_pred)
        rmsle = np.sqrt(msle)

    def linear_regression(self):
        X_train, X_test, y_train, y_test = self.preprocessing(self)
            
        model = LinearRegression()
        model.fit(X_train, y_train)
            
        y_pred = model.predict(X_test)
        self.performance(self, y_test, y_pred)

        self.save_model(self, model)

    def polynomial_regression(self):
        X_train, X_test, y_train, y_test = self.preprocessing(self)
        
        model = PolynomialFeatures(degree=3)
        X_poly = model.fit_transform(X_train)
        model.fit(X_poly, y_train)

        y_pred = model.predict(X_test)
        self.performance(self, y_test, y_pred)

        self.save_model(self, model)
    
    def lasso_regression(self):
        X_train, X_test, y_train, y_test = self.preprocessing(self)
        
        model = Lasso(selection='random', random_state=42)
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_test)
        self.performance(self, y_test, y_pred)

        self.save_model(self, model)

    def decision_tree_regressor(self):
        X_train, X_test, y_train, y_test = self.preprocessing(self)
            
        model = DecisionTreeRegressor(criterion="friedman_mse", random_state=42)
        model.fit(X_train, y_train)
            
        y_pred = model.predict(X_test)
        self.performance(self, y_test, y_pred)

        self.save_model(self, model)

    def random_forest_regressor(self):
        X_train, X_test, y_train, y_test = self.preprocessing(self)
            
        model = RandomForestRegressor(criterion="friedman_mse", random_state=42)
        model.fit(X_train, y_train)
            
        y_pred = model.predict(X_test)
        self.performance(self, y_test, y_pred)

        self.save_model(self, model)

    def gradient_boosting_regressor(self):
        X_train, X_test, y_train, y_test = self.preprocessing(self)
            
        model = GradientBoostingRegressor(loss="huber", criterion="friedman_mse", random_state=42)
        model.fit(X_train, y_train)
            
        y_pred = model.predict(X_test)
        self.performance(self, y_test, y_pred)

        self.save_model(self, model)

class Classification:
    def __init__(self, dataset_path, target_variable, output_path):
        self.dataset_path = dataset_path
        self.target_variable = target_variable
        self.output_path = output_path
    
    def load_dataset(self):
        try:
            dataset = pd.read_csv(self.dataset_path)
            return dataset
        
        except FileNotFoundError:
            print("File {} not found".format(self.dataset_path))
            return None
        
        except:
            print("Some error has occured")
            return None
    
    def preprocessing(self):
        dataset = self.load_dataset()
        if dataset is not None:
            X = dataset.drop(self.target_variable, axis=1)
            y = dataset[self.target_variable]

            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
        
            X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
            return X_train, X_test, y_train, y_test
        else:
            return
    
    def save_model(self, model):
        joblib.dump(model, self.model_path)

    def performance(self, y_test, y_pred):
        cm = confusion_matrix(y_test, y_pred)
        normalized_cm = confusion_matrix(y_test, y_pred, normalized=True)
        acs = accuracy_score(y_test, y_pred)
        bacs = balanced_accuracy_score(y_test, y_pred)
        ps = precision_score(y_test, y_pred)
        aps = average_precision_score(y_test, y_pred)
        rs = recall_score(y_test, y_pred)
        js = jaccard_score(y_test, y_pred)
        f1s = f1_score(y_test, y_pred)
        rocaucs = roc_auc_score(y_test, y_pred)

    def logistic_regression(self):
        X_train, X_test, y_train, y_test = self.preprocessing(self)
            
        model = LogisticRegression(solver="saga", random_state=42)
        model.fit(X_train, y_train)
            
        y_pred = model.predict(X_test)
        self.performance(self, y_test, y_pred)

        self.save_model(self, model)

    def naive_bayes_classification(self):
        X_train, X_test, y_train, y_test = self.preprocessing(self)
            
        model = GaussianNB()
        model.fit(X_train, y_train)
            
        y_pred = model.predict(X_test)
        self.performance(self, y_test, y_pred)

        self.save_model(self, model)

    def gaussian_process_classification(self):
        X_train, X_test, y_train, y_test = self.preprocessing(self)
            
        model = GaussianProcessClassifier(random_state=42)
        model.fit(X_train, y_train)
            
        y_pred = model.predict(X_test)
        self.performance(self, y_test, y_pred)

        self.save_model(self, model)

    def support_vector_classification(self):
        X_train, X_test, y_train, y_test = self.preprocessing(self)
            
        model = SVC(degree=3, kernel="sigmoid",  random_state=42)
        model.fit(X_train, y_train)
            
        y_pred = model.predict(X_test)
        self.performance(self, y_test, y_pred)

        self.save_model(self, model)

    def decision_tree_classification(self):
        X_train, X_test, y_train, y_test = self.preprocessing(self)
            
        model = DecisionTreeClassifier(criterion="entropy", random_state=42)
        model.fit(X_train, y_train)
            
        y_pred = model.predict(X_test)
        self.performance(self, y_test, y_pred)

        self.save_model(self, model)

    def random_forest_classification(self):
        X_train, X_test, y_train, y_test = self.preprocessing(self)
            
        model = RandomForestClassifier(criterion="entropy", random_state=42)
        model.fit(X_train, y_train)
            
        y_pred = model.predict(X_test)
        self.performance(self, y_test, y_pred)

        self.save_model(self, model)
    
    def gradient_boosting_classification(self):
        X_train, X_test, y_train, y_test = self.preprocessing(self)
            
        model = GradientBoostingClassifier(criterion="friedman_mse", random_state=42)
        model.fit(X_train, y_train)
            
        y_pred = model.predict(X_test)
        self.performance(self, y_test, y_pred)

        self.save_model(self, model)
        
class Clustering:
    def __init__(self, dataset_path, output_path):
        self.dataset_path = dataset_path
        self.output_path = output_path
    
    def load_dataset(self):
        try:
            dataset = pd.read_csv(self.dataset_path)
            return dataset
        
        except FileNotFoundError:
            print("File {} not found".format(self.dataset_path))
            return None
        
        except:
            print("Some error has occured")
            return None
        
    def preprocessing(self):
        dataset = self.load_dataset()
        if dataset is not None:
            X = dataset.copy()

            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
        
            return X_scaled
        else:
            return
    
    def save_model(self, model):
        joblib.dump(model, self.model_path)
    
    def k_means_clustering(self):
        X = self.preprocessing(self)
        
        model = KMeans(init="k-means++", algorithm="elkan", random_state=42)
        model.fit(X)
        
        self.save_model(self, model)

        return model.labels_

    def k_medoids_clustering(self):
        X = self.preprocessing(self)
        
        model = KMedoids(init="k-medoids++", random_state=42)
        model.fit(X)
        
        self.save_model(self, model)

        return model.labels_

    def dbscan_clustering(self):
        X = self.preprocessing(self)
        
        model = DBSCAN(metric="manhattan", algorithm="auto")
        model.fit(X)
        
        self.save_model(self, model)

        return model.labels_


    def agglomerative_clustering(self):
        X = self.preprocessing(self)

        model = AgglomerativeClustering(metric="manhattan", linkage="ward")
        model.fit(X)

        self.save_model(self, model)

        return model.labels_

    def gaussian_mixture_clustering(self):
        X = self.preprocessing(self)

        model = GaussianMixture(init_params="k-means++", random_state=42)
        model.fit(X)

        self.save_model(self, model)

        return model.labels_


class FeatureSelection:
    def __init__(self,feature_matrix,target_vector):
        self.X = feature_matrix
        self.y = target_vector
        
    def variance_threshold_selector(self,X):
        selector = VarianceThreshold()
        X_high_variance = selector.fit_transform(X)
        return X_high_variance
    
    def chi_square_selector(self,X, y):
        selector = SelectKBest(chi2)
        X_selected = selector.fit_transform(X, y)
        return X_selected
    
    def mutual_information_selector(self,X, y):
        selector = SelectKBest(mutual_info_classif)
        X_selected = selector.fit_transform(X, y)
        return X_selected
    
    def anova_selector(self,X, y):
        selector = SelectKBest(f_classif)
        X_selected = selector.fit_transform(X, y)
        return X_selected
    
    def lasso_selector(self, X, y):
        model = Lasso()
        model.fit(X, y)
        selected_features = np.where(model.coef_ != 0)[0]
        return X[:, selected_features], selected_features
    
    '''
    def rfe_selector(self,X, y, n_features=5):
        model = RandomForestRegressor(criterion="friedman_mse", random_state=42)
        rfe = RFE(model, n_features_to_select=n_features)
        rfe.fit_transform(X, y)
        return rfe.ranking_
    
    def rfe_selector(self,X, y, n_features=5):
        model = RandomForestClassifier(criterion="entropy", random_state=42)
        rfe = RFE(model, n_features_to_select=n_features)
        rfe.fit_transform(X, y)
        return rfe.ranking_
    
    def efs_selector(self, X, y):
        model = RandomForestRegressor(criterion="friedman_mse", random_state=42)
        efs = ExhaustiveFeatureSelector(model, min_features=1, max_features=X.shape[1], scoring='neg_mean_squared_error', cv=5)
        efs = efs.fit(X, y)
        return efs.best_idx_
    
    def efs_selector(self, X, y):
        model = RandomForestClassifier(criterion="entropy", random_state=42)
        efs = ExhaustiveFeatureSelector(model, min_features=1, max_features=X.shape[1], scoring='neg_mean_squared_error', cv=5)
        efs = efs.fit(X, y)
        return efs.best_idx_
    '''