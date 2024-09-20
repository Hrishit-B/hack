# Importing Required Modules
import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Importing Modules for Regression Models
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.preprocessing import PolynomialFeatures
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error, mean_squared_log_error, r2_score

# Importing Modules for Classification Models
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, balanced_accuracy_score, precision_score, average_precision_score, recall_score, jaccard_score, f1_score, roc_auc_score

# Importing Modules for Clustering Models
from sklearn.metrics import adjusted_mutual_info_score, adjusted_rand_score, calinski_harabasz_score, completeness_score, davies_bouldin_score, fowlkes_mallows_score, homogeneity_score, mutual_info_score, normalized_mutual_info_score, rand_score, silhouette_score, v_measure_score
from sklearn.metrics.cluster import contingency_matrix
from sklearn.cluster import KMeans, DBSCAN, Birch, AffinityPropagation, MeanShift, OPTICS, AgglomerativeClustering
from sklearn.mixture import GaussianMixture

# Defining Regression Class
class Regression:
    
    # Constructor
    def __init__(self, dataset_path, target_variable, output_path):
        self.dataset_path = dataset_path
        self.target_variable = target_variable
        self.output_path = output_path
    
    # Function to read the dataset    
    def load_dataset(self):
        try:
            try:
                dataset = pd.read_csv(self.dataset_path)
            except:
                pass

            try:
                dataset = pd.read_excel(self.dataset_path)
            except:
                pass

            return dataset
        
        except FileNotFoundError:
            print("File {} not found".format(self.dataset_path))
            return None
        
        except:
            print("Some error has occured")
            return None
    
    # Funtion to preprocess the dataset
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
    
    # Function to save results as joblib file
    def save_model(self, model):
        joblib.dump(model, self.model_path)

    # Function to evaluate the performance of the model
    def performance(self, y_test, y_pred):
        r2 = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        mape = mean_absolute_percentage_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        msle = mean_squared_log_error(y_test, y_pred)
        rmsle = np.sqrt(msle)

    # Funtion to train model using linear regression
    def linear_regression(self):
        X_train, X_test, y_train, y_test = self.preprocessing(self)
            
        model = LinearRegression()
        model.fit(X_train, y_train)
            
        y_pred = model.predict(X_test)
        self.performance(self, y_test, y_pred)

        self.save_model(self, model)

    # Function to train the model using polynomial regression
    def polynomial_regression(self):
        X_train, X_test, y_train, y_test = self.preprocessing(self)
        
        model = PolynomialFeatures(degree=3)
        X_poly = model.fit_transform(X_train)
        model.fit(X_poly, y_train)

        y_pred = model.predict(X_test)
        self.performance(self, y_test, y_pred)

        self.save_model(self, model)
    
    # Function to train the model using lasso regression
    def lasso_regression(self):
        X_train, X_test, y_train, y_test = self.preprocessing(self)
        
        model = Lasso(selection='random', random_state=42)
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_test)
        self.performance(self, y_test, y_pred)

        self.save_model(self, model)

    # Function to train the model using a decision tree regressor
    def decision_tree_regressor(self):
        X_train, X_test, y_train, y_test = self.preprocessing(self)
            
        model = DecisionTreeRegressor(criterion="friedman_mse", random_state=42)
        model.fit(X_train, y_train)
            
        y_pred = model.predict(X_test)
        self.performance(self, y_test, y_pred)

        self.save_model(self, model)

    # Function to train the model using a random forest regressor
    def random_forest_regressor(self):
        X_train, X_test, y_train, y_test = self.preprocessing(self)
            
        model = RandomForestRegressor(criterion="friedman_mse", random_state=42)
        model.fit(X_train, y_train)
            
        y_pred = model.predict(X_test)
        self.performance(self, y_test, y_pred)

        self.save_model(self, model)

    # Function to train the model using a gradient boosting regressor
    def gradient_boosting_regressor(self):
        X_train, X_test, y_train, y_test = self.preprocessing(self)
            
        model = GradientBoostingRegressor(loss="huber", criterion="friedman_mse", random_state=42)
        model.fit(X_train, y_train)
            
        y_pred = model.predict(X_test)
        self.performance(self, y_test, y_pred)

        self.save_model(self, model)

# Defining Classification Models Class
class Classification:
    
    # Constructor
    def __init__(self, dataset_path, target_variable, output_path):
        self.dataset_path = dataset_path
        self.target_variable = target_variable
        self.output_path = output_path
    
    # Function to read the dataset
    def load_dataset(self):
        try:
            try:
                dataset = pd.read_csv(self.dataset_path)
            except:
                pass

            try:
                dataset = pd.read_excel(self.dataset_path)
            except:
                pass

            return dataset
        
        except FileNotFoundError:
            print("File {} not found".format(self.dataset_path))
            return None
        
        except:
            print("Some error has occured")
            return None
    
    # Function to preprocess the dataset
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
    
    # Function to save results as joblib file
    def save_model(self, model):
        joblib.dump(model, self.model_path)

    # Function to evaluate the performance of the model
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

    # Function to train model using logistic regression
    def logistic_regression(self):
        X_train, X_test, y_train, y_test = self.preprocessing(self)
            
        model = LogisticRegression(solver="saga", random_state=42)
        model.fit(X_train, y_train)
            
        y_pred = model.predict(X_test)
        self.performance(self, y_test, y_pred)

        self.save_model(self, model)

    # Function to train model using naive bayes classification
    def naive_bayes_classification(self):
        X_train, X_test, y_train, y_test = self.preprocessing(self)
            
        model = GaussianNB()
        model.fit(X_train, y_train)
            
        y_pred = model.predict(X_test)
        self.performance(self, y_test, y_pred)

        self.save_model(self, model)

    # Function to train model using gaussian process classification
    def gaussian_process_classification(self):
        X_train, X_test, y_train, y_test = self.preprocessing(self)
            
        model = GaussianProcessClassifier(random_state=42)
        model.fit(X_train, y_train)
            
        y_pred = model.predict(X_test)
        self.performance(self, y_test, y_pred)

        self.save_model(self, model)

    # Function to train model using support vectorization
    def support_vector_classification(self):
        X_train, X_test, y_train, y_test = self.preprocessing(self)
            
        model = SVC(degree=3, kernel="sigmoid",  random_state=42)
        model.fit(X_train, y_train)
            
        y_pred = model.predict(X_test)
        self.performance(self, y_test, y_pred)

        self.save_model(self, model)

    # Function to train model using decision tree classification
    def decision_tree_classification(self):
        X_train, X_test, y_train, y_test = self.preprocessing(self)
            
        model = DecisionTreeClassifier(criterion="entropy", random_state=42)
        model.fit(X_train, y_train)
            
        y_pred = model.predict(X_test)
        self.performance(self, y_test, y_pred)

        self.save_model(self, model)

    # Function to train model using random forest classification
    def random_forest_classification(self):
        X_train, X_test, y_train, y_test = self.preprocessing(self)
            
        model = RandomForestClassifier(criterion="entropy", random_state=42)
        model.fit(X_train, y_train)
            
        y_pred = model.predict(X_test)
        self.performance(self, y_test, y_pred)

        self.save_model(self, model)
    
    # Function to train model using gradient boosting classification
    def gradient_boosting_classification(self):
        X_train, X_test, y_train, y_test = self.preprocessing(self)
            
        model = GradientBoostingClassifier(criterion="friedman_mse", random_state=42)
        model.fit(X_train, y_train)
            
        y_pred = model.predict(X_test)
        self.performance(self, y_test, y_pred)

        self.save_model(self, model)
        
# Defining Clustering Models Class
class Clusstering:
    
    # Constructor
    def __init__(self, dataset_path, target_variable, output_path):
        self.dataset_path = dataset_path
        self.target_variable = target_variable
        self.output_path = output_path
    
    # Function to read the dataset
    def load_dataset(self):
        try:
            try:
                dataset = pd.read_csv(self.dataset_path)
            except:
                pass

            try:
                dataset = pd.read_excel(self.dataset_path)
            except:
                pass

            return dataset
        
        except FileNotFoundError:
            print("File {} not found".format(self.dataset_path))
            return None
        
        except:
            print("Some error has occured")
            return None
    
    # Funtion to preprocess the dataset
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
    
    # Function to save results as joblib file
    def save_model(self, model):
        joblib.dump(model, self.model_path)

    # Function to evaluate the performance of the model
    def performance(self, y_test, y_pred):
        amis = adjusted_mutual_info_score(y_test, y_pred)
        ars = adjusted_rand_score(y_test, y_pred)
        chs = calinski_harabasz_score(y_test, y_pred)
        cm = contingency_matrix(y_test, y_pred)
        cs = completeness_score(y_test, y_pred)
        dbs = davies_bouldin_score(y_test, y_pred)
        fms = fowlkes_mallows_score(y_test, y_pred)
        hs = homogeneity_score(y_test, y_pred)
        mis = mutual_info_score(y_test, y_pred)
        nmis = normalized_mutual_info_score(y_test, y_pred)
        rs = rand_score(y_test, y_pred)
        ss = silhouette_score(y_test, y_pred)
        vms = v_measure_score(y_test, y_pred)
    
    # Function to train model using k-means clustering
    def kmeans_clustering(self):
        X_train, X_test, y_train, y_test = self.preprocessing(self)
        
        model = KMeans(n_clusters=3, random_state=42)
        model.fit(self.X_test)
        
        y_pred = model.predict(X_test)
        self.performance(self, y_test, y_pred)

        self.save_model(self, model)

    # Function to train model using DBSCAN
    def DBSCAN(self):
        X_train, X_test, y_train, y_test = self.preprocessing(self)
        
        model = DBSCAN(eps=0.5, min_samples=5)
        model.fit(self.X_test)
        
        y_pred = model.labels_
        self.performance(self)

        self.save_model(self, model)
        
    # Function to train model using gaussian mixture model
    def Gaussian_Mixture_Model(self):
        X_train, X_test, y_train, y_test = self.preprocessing(self)
        
        model = GaussianMixture(n_components=3, random_state=42)
        model.fit(self.X_test)
        
        y_pred = model.predict(self.X_test)
        self.performance(self, y_test, y_pred)

        self.save_model(self, model)

    # Funtion to train model using Birch model
    def BIRCH(self):
        X_train, X_test, y_train, y_test = self.preprocessing(self)
        
        model = Birch(n_clusters=3, threshold=0.5, compute_labels=True)
        model.fit(self.X_test)
        
        y_pred = model.labels_
        self.performance(self, y_test, y_pred)

        self.save_model(self, model)

    # Function to train model using affinity propagation
    def Affinity_Propagation(self):
        X_train, X_test, y_train, y_test = self.preprocessing(self)
        
        model = AffinityPropagation(damping=0.5, max_iter=200, convergence_iter=15, random_state=42)
        model.fit(self.X_test)
        
        y_pred = model.labels_
        self.performance(self, y_test, y_pred)

        self.save_model(self, model)

    # Funtion to train the model using mean shift algorithm
    def Mean_Shift(self):
        X_train, X_test, y_train, y_test = self.preprocessing(self)
        
        model = MeanShift(bandwidth=0.5, bin_seeding=True, cluster_all=True)
        model.fit(self.X_test)
        
        y_pred = model.labels_
        self.performance(self, y_test, y_pred)

        self.save_model(self, model)

    # Function to train model with OPTICS model
    def OPTICS(self):
        X_train, X_test, y_train, y_test = self.preprocessing(self)
        
        model = OPTICS(eps=0.5, min_samples=5, xi=0.05, min_cluster_size=2)
        model.fit(self.X_test)
        
        y_pred = model.labels_
        self.performance(self, y_test, y_pred)

        self.save_model(self, model)
    
    # Funtion to train model with agglomerative hierarcy algorithm
    def Agglomerative_Hierarchy(self):
        X_train, X_test, y_train, y_test = self.preprocessing(self)
        
        model = AgglomerativeClustering(n_clusters=3, affinity='euclidean', linkage='ward')
        model.fit(self.X_test)
        
        y_pred = model.labels_
        self.performance(self, y_test, y_pred)

        self.save_model(self, model)
