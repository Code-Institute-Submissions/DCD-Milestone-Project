from flask import Flask, render_template, redirect, request, url_for, send_file, session, request, flash, Blueprint
from flask_login import LoginManager,login_user, login_required, logout_user, current_user, UserMixin, AnonymousUserMixin

## PLOT DISPLAY

from io import BytesIO
import base64


## DATA PACKAGES

import numpy as np
import matplotlib
matplotlib.use('Agg')                                                           ## allowing plots to be displayed without opening new window
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

## PREPROCESSING

from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix

## CLASSIFICATION

from sklearn.neighbors import KNeighborsClassifier
from matplotlib.colors import ListedColormap
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

## REGRESSION

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor

## CLUSTERING

import scipy.cluster.hierarchy as sch
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import KMeans


## BLUEPRINT INIT

algorithms_blueprint = Blueprint(
    'algorithms', __name__,
    template_folder="templates"
    )
    
from .models import *


#################################################################### ML ESTIMATOR TYPES SUMMARY #####################################################################################################################    
    
    
## ML TYPE CHOICE HANDLING ##########################################################     
    
@algorithms_blueprint.route('/summary/<type_id>')
def summary(type_id):
    if type_id == "Regression":
        return redirect(url_for("algorithms.regression"))
    elif type_id == "Classification": 
        return redirect(url_for("algorithms.classification"))
    else:
        return redirect(url_for("algorithms.clustering"))
        
## REGRESSION INITIATION ############################################################

@algorithms_blueprint.route('/regression')
def regression():
    regressors = Regression.query.all()
    algo = AlgoTypes.query.filter_by(algotype_id = 1).first()
    
    ## Describe dataset
    
    dataset = pd.read_csv('Position_Salaries.csv')
    X = dataset.iloc[:, 1:2].values
    y = dataset.iloc[:, 2:3].values
    describe = dataset.describe()
    rows = len(dataset.index)
    columns = len(dataset.columns)
    pred = 'Choose Algorithm'
    
    return render_template('regression.html', data = dataset.to_html(),  
                            describe = describe.to_html(), pred = pred, rows = rows, 
                            columns = columns, regressors = regressors, algo = algo, 
                            user = current_user.username)
   
## REGRESSION ALGORITHM CHOICE HANDLING ############################################
    
@algorithms_blueprint.route('/regressor/<regressor_id>')
def regressor(regressor_id):
    regressors = Regression.query.all() 
    algo = AlgoTypes.query.filter_by(algotype_id = 1).first()
    
    ## Describe dataset
    dataset = pd.read_csv('Position_Salaries.csv')
    X = dataset.iloc[:, 1:2].values
    y = dataset.iloc[:, 2:3].values
    describe = dataset.describe()
    rows = len(dataset.index)
    columns = len(dataset.columns)
    pred = 'Choose Algorithm'
    choice = regressor_id
    plt.gcf().clear()
    
    if choice == '1':
        ## Fitting Polynomial Regression to the dataset
        poly_reg = PolynomialFeatures(degree = 4)                               ## Transform X into new matrix composed with X^4
        X_poly = poly_reg.fit_transform(X)                                      ## First fit, then transform the object into poly - used in a plot later        
        lin_reg = LinearRegression()
        lin_reg.fit(X_poly, y)
        
        ## Visualising The Polynomial Regression
        img = BytesIO()                                               ## initiate an image for holding the plot
        plt.gcf().clear()                                                       ## clearing previous plots from canvas
        X_grid = np.arange(min(X), max(X), 0.1)   
        X_grid = X_grid.reshape(len(X_grid),1)  
        sns.set_style("darkgrid", {"axes.facecolor": ".9"})
        plt.scatter(X,y, color = 'red')
        plt.plot(X_grid, lin_reg.predict(poly_reg.fit_transform(X_grid)),color ='darkblue')    
        plt.ylim(ymin=0)
        plt.title('Salary Estimate - Polynomial Regression')
        plt.xlabel('Position Level')
        plt.ylabel('Salary', fontsize=12)
        plt.yticks(fontsize=10)
        
        plt.savefig(img, format='png')                                          ## save generated plot as an image    
        img.seek(0)
        plot_url = base64.b64encode(img.getvalue())
        plt.gcf().clear()
        
        ## Predicting result with Polynomial Regression
        pred = lin_reg.predict(poly_reg.fit_transform(6.5))
        pred = pred.flat[0]
        pred = ('%.2f' % (pred,)).rstrip('0').rstrip('.')
        
    elif choice == '2':
        # Feature Scaling - needed for rbf SVR
        sc_X = StandardScaler()
        X = sc_X.fit_transform(X)
        sc_y = StandardScaler()
        y = sc_y.fit_transform(y)
        
        # Fitting SVR to the dataset
        regressor = SVR(kernel = 'rbf')                                         ## Kernel choice - linear, poly or gaussian(rbf)
        regressor.fit(X,y)
        
        img = BytesIO()
        X_grid = np.arange(min(X), max(X), 0.1)
        X_grid = X_grid.reshape((len(X_grid), 1))
        plt.scatter(X, y, color = 'red')
        sns.set_style("darkgrid", {"axes.facecolor": ".9"})
        plt.plot(X_grid, regressor.predict(X_grid), color = 'darkblue')
        plt.title('Salary Estimate - SVR')
        plt.xlabel('Position level')
        plt.ylabel('Salary', fontsize=12)
        plt.yticks(fontsize=10)
        plt.savefig(img, format='png')
        img.seek(0)
        plot_url = base64.b64encode(img.getvalue())
        plt.gcf().clear()
        
        # Predicting result - has to be changed as scaling was applied 
        pred = sc_y.inverse_transform(regressor.predict(sc_X.transform(np.array([[6.5]]))))     ## transform value into array with numpy.array
        pred = pred.flat[0]
        pred = ('%.2f' % (pred,)).rstrip('0').rstrip('.')                       ## for better display
    else: 
        
        # Fitting Random Forest Regression Model to the dataset
        regressor = RandomForestRegressor(n_estimators = 600, random_state = 0)
        regressor.fit(X,y)
        
        # Visualising the Regression results (for higher resolution and smoother curve)
        img = BytesIO()
        X_grid = np.arange(min(X), max(X), 0.01)                                ## increase the resolution
        X_grid = X_grid.reshape((len(X_grid), 1))   
        plt.scatter(X, y, color = 'red')
        sns.set_style("darkgrid", {"axes.facecolor": ".9"})
        plt.plot(X_grid, regressor.predict(X_grid), color = 'darkblue')
        plt.ylim(ymin=0)
        plt.title('Salary Estimate - RandomForest Regression')
        plt.xlabel('Position level')
        plt.ylabel('Salary', fontsize=12)
        plt.yticks(fontsize=10)
        plt.savefig(img, format='png')
        img.seek(0)
        plot_url = base64.b64encode(img.getvalue())
        plt.gcf().clear()
        
        # Predicting result
        pred = regressor.predict(6.5)
        pred = pred.flat[0]
        pred = ('%.2f' % (pred,)).rstrip('0').rstrip('.')
        
    return render_template('regression.html', data = dataset.to_html(), 
                            describe = describe.to_html(), pred = pred, 
                            plot_url=plot_url, rows = rows, columns = columns, 
                            regressors = regressors, algo = algo, user = current_user.username)

        
## CLASSIFICATION INITIATION ########################################################

@algorithms_blueprint.route('/classification')
def classification():
    classifiers = Classification.query.all()
    algo = AlgoTypes.query.filter_by(algotype_id = 2).first()
    
    ## Describe dataset
   
    dataset = pd.read_csv('Social_Network_Ads.csv')
    dataset_head = dataset.head(10)
    stats_data = dataset.iloc[:,2:4]
    describe = stats_data.describe()
    rows = len(dataset.index)
    columns = len(dataset.columns)
    X = dataset.iloc[:, [2, 3]].values
    y = dataset.iloc[:, 4].values  
    pred = 'Choose Algorithm'
    
    return render_template('classification.html', data = dataset_head.to_html(),  
                            describe = describe.to_html(), pred = pred, rows = rows, 
                            columns = columns, classifiers = classifiers, 
                            algo = algo, user = current_user.username) 

## CLASSIFICATION ALGORITHM CHOICE HANDLING #########################################

@algorithms_blueprint.route('/classifier/<classifier_id>')
def classifier(classifier_id):
    classifiers = Classification.query.all()
    algo = AlgoTypes.query.filter_by(algotype_id = 2).first()
    
    ## Describe dataset
    dataset = pd.read_csv('Social_Network_Ads.csv')
    dataset_head = dataset.head(10)
    stats_data = dataset.iloc[:,2:4]
    describe = stats_data.describe()
    rows = len(dataset.index)
    columns = len(dataset.columns)
    X = dataset.iloc[:, [2, 3]].values
    y = dataset.iloc[:, 4].values  
    plt.gcf().clear()
    
    # Splitting the dataset into the Training set and Test set
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)
    
    # Feature Scaling
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)
    
    # Choose classifier to fit the Training set
    classifier = GaussianNB() if classifier_id == '3'  \
                              else KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2) if classifier_id == '1' \
                              else SVC(kernel = 'rbf', random_state = 0 )
    classifier.fit(X_train, y_train)
    
    # Predicting the Test set results
    y_pred = classifier.predict(X_test)
    
    # Making the Confusion Matrix
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_test, y_pred)
    df = pd.DataFrame(cm, index = ['Bought: Yes', 'Bought: No'], columns = ['Predicted: Yes', 'Predicted: No'])
    accuracy = (df.iloc[0,0] + df.iloc[1,1])
    accuracy =  "{}%".format(float(accuracy))
    
    
    # Visualising the Test set results with the heatmap
    from matplotlib.colors import ListedColormap
    img = BytesIO()
    
    X_set, y_set = X_test, y_test
    X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                         np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
    plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
                 alpha = 0.85, cmap = ListedColormap(('darkred', 'darkgreen')))
    plt.xlim(X1.min(), X1.max())
    plt.ylim(X2.min(), X2.max())
    for i, j in enumerate(np.unique(y_set)):
        plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                    c = ListedColormap(('red', 'green'))(i), label = j)
    plt.title('Test Set Results')
    plt.xlabel('Age')
    plt.ylabel('Estimated Salary')
    plt.legend()
    plt.savefig(img, format='png')
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue())
    plt.gcf().clear()
    
    
    
    return render_template('classification.html', data = dataset_head.to_html(), 
                            describe = describe.to_html(), cma = df.to_html(), 
                            plot_url=plot_url, rows = rows, columns = columns, 
                            classifiers = classifiers, acc = accuracy, 
                            algo = algo, user = current_user.username)
    

## CLUSTERING INITIATION ############################################################

@algorithms_blueprint.route('/clustering')
def clustering():
    clusterers = Clustering.query.all()
    algo = AlgoTypes.query.filter_by(algotype_id = 3).first()
    
    ## Describe dataset
    dataset = pd.read_csv('Mall_Customers.csv')
    X = dataset.iloc[:, [3, 4]].values
    dataset_head = dataset.head(10)
    stats_data = dataset.iloc[:,2:5]
    describe = stats_data.describe()
    rows = len(dataset.index)
    columns = len(dataset.columns)
    pred = 'Choose Algorithm'
    
    return render_template('clustering.html', data = dataset_head.to_html(),  
                            describe = describe.to_html(), pred = pred, 
                            rows = rows, columns = columns, 
                            clusterers = clusterers, algo = algo, 
                            user = current_user.username)
   
## CLUSTERING ALGORITHM CHOICE HANDLING #############################################  
    
@algorithms_blueprint.route('/clusterer/<clusterer_id>')
def clusterer(clusterer_id):
    clusterers = Clustering.query.all()
    algo = AlgoTypes.query.filter_by(algotype_id = 3).first()
    
    ## Describe dataset
    dataset = pd.read_csv('Mall_Customers.csv')
    X = dataset.iloc[:, [3, 4]].values
    dataset_head = dataset.head(10)
    stats_data = dataset.iloc[:,2:5]
    describe = stats_data.describe()
    rows = len(dataset.index)
    columns = len(dataset.columns)
    pred = 'Choose Algorithm'
    
    choice = clusterer_id
    
    if choice == '2':
        
        ## Using the dendogram to find the optimal number of clusters
        plt.gcf().clear()
        dendrogram = sch.dendrogram(sch.linkage(X,method = 'ward'))
        img_dendrogram = BytesIO()
        sns.set_style("darkgrid", {"axes.facecolor": ".9"})
        plt.title("Dendrogram")
        plt.xlabel('Customers')
        plt.ylabel('Euclidean distances')
        plt.savefig(img_dendrogram, format='png')
        img_dendrogram.seek(0)
        plot_determine = base64.b64encode(img_dendrogram.getvalue())
        ## Result = 5
        
        ## Fitting Hierarchical Clustering to the dataset (optimal clusters = 5)
        hc = AgglomerativeClustering(n_clusters = 5, affinity = 'euclidean', linkage = 'ward')
        y_hc = hc.fit_predict(X)
        
        ## Visualising the clusters 
        img = BytesIO()
        plt.gcf().clear()
        plt.scatter(X[y_hc == 0, 0], X[y_hc == 0, 1],   ## specify that we want first cluster + first column vs second column for 'y'
            s = 100, c = 'red',label = 'Savers')                            ## size for datapoints/color
        plt.scatter(X[y_hc == 1, 0], X[y_hc == 1, 1],s = 100, c = 'blue',label = 'Average') 
        plt.scatter(X[y_hc == 2, 0], X[y_hc == 2, 1],s = 100, c = 'green',label = 'Target Group') 
        plt.scatter(X[y_hc == 3, 0], X[y_hc == 3, 1],s = 100, c = 'orange',label = 'Overspenders') 
        plt.scatter(X[y_hc == 4, 0], X[y_hc == 4, 1],s = 100, c = 'magenta',label = 'Careful') 
        sns.set_style("darkgrid", {"axes.facecolor": ".9"})
        plt.title('Suggested Clusters')
        plt.xlabel('Annual income (k$)')
        plt.ylabel('Spending Score (1-100)', fontsize=12)
        plt.ylim(ymin=0)
        plt.legend(fontsize = 9)
        plt.savefig(img, format='png')
        img.seek(0)
        plot_url = base64.b64encode(img.getvalue())
        
    if choice == '1':    
        
        ## Using the elbow method to find the optimal number of clusters
        wcss = []    ## initialize the list
        for i in range(1, 11):
            kmeans = KMeans(n_clusters = i,        ## from 1 to 10
                            init = 'k-means++',    ## k-means++ to avoid random initialziation trap
                            max_iter = 300,        ## 300 is deafault        
                            n_init  = 10,          ## algorithm runs with different initial centroids      
                            random_state = 0)
            kmeans.fit(X)
            wcss.append(kmeans.inertia_)           ## to compute wcss   
        ## Result = 5    
        
        ## Visualising Elbow Method
        plt.gcf().clear()
        img_elbow = BytesIO()
        plt.plot(range(1,11), wcss)
        plt.title('The Elbow Method')
        plt.xlabel('Number of clusters')
        plt.ylabel('WCSS')
        sns.set_style("darkgrid", {"axes.facecolor": ".9"})
        plt.savefig(img_elbow, format='png')
        img_elbow.seek(0)
        plot_determine = base64.b64encode(img_elbow.getvalue())
        
        
        
        ## Applying k-means to the mall dataset - from the plot we can see that optimum is 5 clusters.
        kmeans = KMeans(n_clusters = 5,
                        init = 'k-means++',    ## k-means++ to avoid random initialziation trap
                        max_iter = 300,        ## 300 is deafault        
                        n_init  = 10,          ## algorithm runs with different initial centroids      
                        random_state = 0)
        y_kmeans = kmeans.fit_predict(X)       ## fit_predict returns a cluster for each observation 
        
        ## Visualising the clusters
        img = BytesIO()
        plt.gcf().clear()
        sns.set_style("darkgrid", {"axes.facecolor": ".9"})
        plt.scatter(X[y_kmeans == 0, 0], X[y_kmeans == 0, 1],   ## specify that we want first cluster + first column vs second column for 'y'
                    s = 100, c = 'red',label = 'Savers')                            ## size for datapoints/color
        plt.scatter(X[y_kmeans == 1, 0], X[y_kmeans == 1, 1],s = 100, c = 'blue',label = 'Average') 
        plt.scatter(X[y_kmeans == 2, 0], X[y_kmeans == 2, 1],s = 100, c = 'green',label = 'Target Group') 
        plt.scatter(X[y_kmeans == 3, 0], X[y_kmeans == 3, 1],s = 100, c = 'orange',label = 'Overspenders') 
        plt.scatter(X[y_kmeans == 4, 0], X[y_kmeans == 4, 1],s = 100, c = 'magenta',label = 'Careful') 
        plt.scatter(kmeans.cluster_centers_[:,0],kmeans.cluster_centers_[:,1],         ## cluster centers coordinates
                    s = 200, c = 'black', label = 'Centroids')
        plt.title('Suggested Clusters')
        plt.xlabel('Annual income (k$)')
        plt.ylabel('Spending Score (1-100)', fontsize=12)
        plt.ylim(ymin=0)
        plt.legend(fontsize = 9)
        plt.savefig(img, format='png')
        img.seek(0)
        plot_url = base64.b64encode(img.getvalue())
        
        
    return render_template('clustering.html', data = dataset_head.to_html(), 
                            describe = describe.to_html(), plot_determine = plot_determine, 
                            plot_url=plot_url, rows = rows, columns = columns, 
                            clusterers = clusterers, algo = algo, 
                            user = current_user.username)
    