## APP
import os
from flask import Flask, render_template, redirect, request, url_for, send_file, session
from flask_sqlalchemy import SQLAlchemy
import flask_whooshalchemy as wa
from io import BytesIO
import StringIO
import base64

## FORMS 
from flask_wtf import FlaskForm
from flask_wtf.file import FileField, FileRequired
from werkzeug.utils import secure_filename
from wtforms import StringField
from wtforms.validators import InputRequired,  DataRequired

## DATA
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt, mpld3
import pandas as pd
import seaborn as sns

## ML
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from ml_modules import *


app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///database.db'
app.config['SECRET_KEY'] = 'BiggestSecret'
db = SQLAlchemy(app)


class CodeRepo(db.Model):
    __searchable__ = ['name', 'type_of_algorithm','complexity','method','author']
    __tablename__ = 'CodeRepo'
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(300))
    type_of_algorithm = db.Column(db.String(300))
    complexity = db.Column(db.String(300))
    method = db.Column(db.String(300))
    author = db.Column(db.String(300))
    file = db.Column(db.LargeBinary)
    downloads = db.Column(db.Integer)
    
    # def __init__(self, id, name, type_of_algorithm, complexity, method, author, downloads):
    #     self.id = id
    #     self.name = name
    #     self.type_of_algorithm = type_of_algorithm
    #     self.complexity = complexity
    #     self.method = method
    #     self.author = author
    #     self.downloads = downloads
        


class Types(db.Model):
    __tablename__ = 'Types'
    id = db.Column(db.Integer, primary_key=True)
    type_of_algorithm = db.Column(db.String(300))

class Complexities(db.Model):
    __tablename__ = 'Complexity'
    id = db.Column(db.Integer, primary_key=True)
    complexity = db.Column(db.String(300))
    

class Methods(db.Model):    
    __tablename__ = 'Methods'
    id = db.Column(db.Integer, primary_key=True)
    method = db.Column(db.String(300))
    


class Regression(db.Model):
    __tablename__ = 'Regression'
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(300))
    

class Classification(db.Model):
    __tablename__ = 'Classification'
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(300))
    

class Clustering(db.Model):
    __tablename__ = 'Clustering'
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(300))
    
    
Assembler = db.Table('Assembler',
    db.Column('algotype_id', db.Integer, db.ForeignKey('AlgoTypes.algotype_id')),
    db.Column('preprocess_id', db.Integer, db.ForeignKey('Preprocessing.preprocess_id')),
    db.Column('issue_id', db.Integer, db.ForeignKey('RedFlags.issue_id'))
)       

class AlgoTypes(db.Model):
    __tablename__ = 'AlgoTypes'
    algotype_id = db.Column(db.Integer, primary_key=True)
    algo_type = db.Column(db.String(300))
    preprocessers = db.relationship('Preprocessing', secondary=Assembler, backref=db.backref('algos', lazy = 'dynamic'))
    issues = db.relationship('RedFlags', secondary=Assembler, backref=db.backref('algos', lazy = 'dynamic'))
    
    
class Preprocessing(db.Model):
    __tablename__ = 'Preprocessing'
    preprocess_id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(300))    
    
class RedFlags(db.Model):
    __tablename__ = 'RedFlags'
    issue_id = db.Column(db.Integer, primary_key=True)
    issue = db.Column(db.String(300))         
    
   
wa.whoosh_index(app, CodeRepo)     
      
@app.route('/')
@app.route('/library')
def library():
    codes = CodeRepo.query.all()
 
    return render_template("library.html", codes = codes)
    
    
@app.route('/download_code/<code_id>', methods=['GET'])
def download_code(code_id):
    the_code = CodeRepo.query.filter_by(id = code_id).first()
    click = the_code.downloads + 1
    the_code.downloads = click
    db.session.commit()
    
    return send_file(BytesIO(the_code.file), attachment_filename='{}.pdf'.format(the_code.name), as_attachment=True)
    
@app.route('/add_request')
def add_request():
    types = Types.query.all()
    complexities = Complexities.query.all()
    methods = Methods.query.all()
    
    
    return render_template ('add_request.html', types = types, complexities = complexities, methods = methods)
    
    
@app.route('/new_code', methods = ['POST'])
def new_code():
    codes = CodeRepo.query.all()
    ## fix author
    if not 'inputFile' in request.files:
        return render_template ('bad.html')
    if not 'type_of_algorithm' in request.form or not 'complexity' in request.form:
        return render_template ('bad.html')    
    if not 'method' in request.form:
        return render_template ('bad.html')
    if request.form['name'] == '' or request.form['author'] == '':
        return render_template ('bad.html')
    
    else:    
        code_file = request.files['inputFile']
        code_file = code_file.read()

        code = CodeRepo(name=request.form['name'],
                        type_of_algorithm=request.form['type_of_algorithm'],
                        complexity=request.form['complexity'],
                        method=request.form['method'],
                        author=request.form['author'],
                        file = code_file,
                        downloads = 0)
        db.session.add(code)
        db.session.commit()
        return redirect(url_for("library"))

    
@app.route('/edit_code/<code_id>')
def edit_code(code_id):
    the_code = CodeRepo.query.filter_by(id = code_id).first()
    types = Types.query.all()
    complexities = Complexities.query.all()
    methods = Methods.query.all()
    return render_template('edit_code.html', code = the_code, types = types, complexities = complexities, methods = methods)    
    
    
@app.route('/update_code/<code_id>', methods=["POST"])
def update_code(code_id):
    the_code = CodeRepo.query.filter_by(id = code_id).first()

    the_code.name=request.form['name']
    the_code.type_of_algorithm=request.form['type_of_algorithm']
    the_code.complexity=request.form['complexity']
    the_code.method=request.form['method']
    the_code.author=request.form['author']
    if request.form['name'] == '' or request.form['author'] == '':
        return render_template ('bad.html')
    if 'inputFile' in request.files:
        code_file = request.files['inputFile']
        code_file = code_file.read()
        the_code.file = code_file
        db.session.commit()
        return redirect(url_for("library"))

    else:    
        db.session.commit()
        return redirect(url_for("library"))
    
    
@app.route('/delete_code/<code_id>')
def delete_code(code_id):
    the_code = CodeRepo.query.filter_by(id = code_id).first()
    db.session.delete(the_code)
    db.session.commit()
    return redirect(url_for("library"))
    
    
@app.route('/summary/<type_id>')
def summary(type_id):
    if type_id == "Regression":
        return redirect(url_for("regression"))
    elif type_id == "Classification": 
        return redirect(url_for("classification"))
    else:
        return redirect(url_for("clustering"))



@app.route('/classification')
def classification():
    classifiers = Classification.query.all()
    algo = AlgoTypes.query.filter_by(algotype_id = 2).first()
    ## Dataset Variables
   
    dataset = pd.read_csv('Social_Network_Ads.csv')
    dataset_head = dataset.head()
    stats_data = dataset.iloc[:,2:4]
    describe = stats_data.describe()
    rows = len(dataset.index)
    columns = len(dataset.columns)
    X = dataset.iloc[:, [2, 3]].values
    y = dataset.iloc[:, 4].values  
    pred = '6.5'
    
    return render_template('classification.html', data = dataset_head.to_html(),  describe = describe.to_html(), pred = pred, rows = rows, columns = columns, classifiers = classifiers, algo = algo) 

@app.route('/classifier/<classifier_id>')
def classifier(classifier_id):
    classifiers = Classification.query.all()
    
    # Importing the dataset
    dataset = pd.read_csv('Social_Network_Ads.csv')
    dataset_head = dataset.head()
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
    
    # Fitting classifier to the Training set
    
   
    classifier = GaussianNB() if classifier_id == '1' else KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2) if classifier_id == '2' else SVC(kernel = 'rbf', random_state = 0 )
    classifier.fit(X_train, y_train)
    
    # Predicting the Test set results
    y_pred = classifier.predict(X_test)
    
    # Making the Confusion Matrix
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_test, y_pred)
    df = pd.DataFrame(cm, index = ['Actual: Yes', 'Actual: No'], columns = ['Predicted: Yes', 'Predicted: No'])
    accuracy = (df.iloc[0,0] + df.iloc[1,1])
    accuracy =  "{}%".format(float(accuracy))
    
    
    # Visualising the Training set results
    from matplotlib.colors import ListedColormap
    img = StringIO.StringIO()
    
    X_set, y_set = X_train, y_train
    X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                         np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
    plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
                 alpha = 0.75, cmap = ListedColormap(('red', 'green')))
    plt.xlim(X1.min(), X1.max())
    plt.ylim(X2.min(), X2.max())
    for i, j in enumerate(np.unique(y_set)):
        plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                    c = ListedColormap(('red', 'green'))(i), label = j)
    plt.title('Naive Bayes (Training set)')
    plt.xlabel('Age')
    plt.ylabel('Estimated Salary')
    plt.legend()
    plt.savefig(img, format='png')
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue())
    plt.gcf().clear()
    
    return render_template('classification.html', data = dataset_head.to_html(), describe = describe.to_html(), cma = df.to_html(), plot_url=plot_url, rows = rows, columns = columns, classifiers = classifiers, acc = accuracy)
    

@app.route('/regression')
def regression():
    regressors = Regression.query.all()
    algo = AlgoTypes.query.filter_by(algotype_id = 1).first()
    dataset = pd.read_csv('Position_Salaries.csv')
    X = dataset.iloc[:, 1:2].values
    y = dataset.iloc[:, 2:3].values
    dataset_head = dataset.head()
    describe = dataset.describe()
    rows = len(dataset.index)
    columns = len(dataset.columns)
    pred = '6.5'
    
    return render_template('regression.html', data = dataset_head.to_html(),  describe = describe.to_html(), pred = pred, rows = rows, columns = columns, regressors = regressors, algo = algo)
    
@app.route('/regressor/<regressor_id>')
def regressor(regressor_id):
    regressors = Regression.query.all() 
    dataset = pd.read_csv('Position_Salaries.csv')
    X = dataset.iloc[:, 1:2].values
    y = dataset.iloc[:, 2:3].values
    dataset_head = dataset.head()
    describe = dataset.describe()
    rows = len(dataset.index)
    columns = len(dataset.columns)
    pred = '6.5'
    choice = regressor_id
    plt.gcf().clear()
    
    if choice == '1':
        poly_reg = PolynomialFeatures(degree = 4)          
        X_poly = poly_reg.fit_transform(X)                     
        lin_reg = LinearRegression()
        lin_reg.fit(X_poly, y)
        
        img = StringIO.StringIO()
        plt.gcf().clear()
        X_grid = np.arange(min(X), max(X), 0.1)   
        X_grid = X_grid.reshape(len(X_grid),1)         
        plt.scatter(X,y, color = 'red')
        plt.plot(X_grid, lin_reg.predict(poly_reg.fit_transform(X_grid)), color = 'blue')    
        plt.title('Reality Check (Polynomial Regression)')
        plt.xlabel('Position Level')
        plt.ylabel('Salary')
        plt.savefig(img, format='png')
        img.seek(0)
        plot_url = base64.b64encode(img.getvalue())
        plt.gcf().clear()
        
        pred = lin_reg.predict(poly_reg.fit_transform(6.5))
        
    elif choice == '2':
        sc_X = StandardScaler()
        X = sc_X.fit_transform(X)
        sc_y = StandardScaler()
        y = sc_y.fit_transform(y)
        
        regressor = SVR(kernel = 'rbf')
        regressor.fit(X,y)
        
        img = StringIO.StringIO()
        X_grid = np.arange(min(X), max(X), 0.1)
        X_grid = X_grid.reshape((len(X_grid), 1))
        plt.scatter(X, y, color = 'red')
        plt.plot(X_grid, regressor.predict(X_grid), color = 'blue')
        plt.title('Truth or Bluff (Regression Model)')
        plt.xlabel('Position level')
        plt.ylabel('Salary')
        plt.savefig(img, format='png')
        img.seek(0)
        plot_url = base64.b64encode(img.getvalue())
        plt.gcf().clear()
        
        pred = sc_y.inverse_transform(regressor.predict(sc_X.transform(np.array([[6.5]]))))
        
    else: 
        regressor = RandomForestRegressor(n_estimators = 600, random_state = 0)
        regressor.fit(X,y)
        
        img = StringIO.StringIO()
        X_grid = np.arange(min(X), max(X), 0.01)
        X_grid = X_grid.reshape((len(X_grid), 1))
        plt.scatter(X, y, color = 'red')
        plt.grid()
        plt.plot(X_grid, regressor.predict(X_grid), color = 'blue')
        plt.title('Reality Check (Random Forest Regression Model)')
        plt.xlabel('Position level')
        plt.ylabel('Salary')
        plt.savefig(img, format='png')
        img.seek(0)
        plot_url = base64.b64encode(img.getvalue())
        plt.gcf().clear()
        pred = regressor.predict(6.5)
        
    return render_template('regression.html', data = dataset_head.to_html(), describe = describe.to_html(), pred = pred, plot_url=plot_url, rows = rows, columns = columns, regressors = regressors)


@app.route('/clustering')
def clustering():
    clusterers = Clustering.query.all()
    algo = AlgoTypes.query.filter_by(algotype_id = 3).first()
    dataset = pd.read_csv('Mall_Customers.csv')
    X = dataset.iloc[:, [3, 4]].values
    dataset_head = dataset.head()
    stats_data = dataset.iloc[:,2:5]
    describe = stats_data.describe()
    rows = len(dataset.index)
    columns = len(dataset.columns)
    pred = '6.5'
    
    return render_template('clustering.html', data = dataset_head.to_html(),  describe = describe.to_html(), pred = pred, rows = rows, columns = columns, clusterers = clusterers, algo = algo)
    
@app.route('/clusterer/<clusterer_id>')
def clusterer(clusterer_id):
    clusterers = Clustering.query.all()
    algo = AlgoTypes.query.filter_by(algotype_id = 3).first()
    dataset = pd.read_csv('Mall_Customers.csv')
    X = dataset.iloc[:, [3, 4]].values
    dataset_head = dataset.head()
    stats_data = dataset.iloc[:,2:5]
    describe = stats_data.describe()
    rows = len(dataset.index)
    columns = len(dataset.columns)
    pred = '6.5'
    
    choice = clusterer_id
    
    if choice == '1':
        
        plt.gcf().clear()
        dendrogram = sch.dendrogram(sch.linkage(X,method = 'ward'))
        img_dendrogram = StringIO.StringIO()
        plt.title("Dendrogram")
        plt.xlabel('Customers')
        plt.ylabel('Euclidean distances')
        plt.savefig(img_dendrogram, format='png')
        img_dendrogram.seek(0)
        plot_determine = base64.b64encode(img_dendrogram.getvalue())
        
        
        
        hc = AgglomerativeClustering(n_clusters = 5, affinity = 'euclidean', linkage = 'ward')
        y_hc = hc.fit_predict(X)
        
        img = StringIO.StringIO()
        plt.gcf().clear()
        plt.scatter(X[y_hc == 0, 0], X[y_hc == 0, 1],   ## specify that we want first cluster + first column vs second column for 'y'
            s = 100, c = 'red',label = 'Careful')                            ## size for datapoints/color
        plt.scatter(X[y_hc == 1, 0], X[y_hc == 1, 1],s = 100, c = 'blue',label = 'Standard') 
        plt.scatter(X[y_hc == 2, 0], X[y_hc == 2, 1],s = 100, c = 'green',label = 'Target') 
        plt.scatter(X[y_hc == 3, 0], X[y_hc == 3, 1],s = 100, c = 'cyan',label = 'Careless') 
        plt.scatter(X[y_hc == 4, 0], X[y_hc == 4, 1],s = 100, c = 'magenta',label = 'Sensible') 
        plt.title('Clusters of clients')
        plt.xlabel('Annual income (k$)')
        plt.ylabel('Spending Score (1-100)')
        plt.legend()
        plt.savefig(img, format='png')
        img.seek(0)
        plot_url = base64.b64encode(img.getvalue())
        
    if choice == '2':    
        
        wcss = []    ## initialize the list
        for i in range(1, 11):
            kmeans = KMeans(n_clusters = i,        ## from 1 to 10
                            init = 'k-means++',    ## k-means++ to avoid random initialziation trap
                            max_iter = 300,        ## 300 is deafault        
                            n_init  = 10,          ## algorithm runs with different initial centroids      
                            random_state = 0)
            kmeans.fit(X)
            wcss.append(kmeans.inertia_)           ## to compute wcss   
        
        plt.gcf().clear()
        img_elbow = StringIO.StringIO()
        plt.plot(range(1,11), wcss)
        plt.title('The Elbow Method')
        plt.xlabel('Number of clusters')
        plt.ylabel('WCSS')
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
        img = StringIO.StringIO()
        plt.gcf().clear()
        plt.scatter(X[y_kmeans == 0, 0], X[y_kmeans == 0, 1],   ## specify that we want first cluster + first column vs second column for 'y'
                    s = 100, c = 'red',label = 'Careful')                            ## size for datapoints/color
        plt.scatter(X[y_kmeans == 1, 0], X[y_kmeans == 1, 1],s = 100, c = 'blue',label = 'Standard') 
        plt.scatter(X[y_kmeans == 2, 0], X[y_kmeans == 2, 1],s = 100, c = 'green',label = 'Target') 
        plt.scatter(X[y_kmeans == 3, 0], X[y_kmeans == 3, 1],s = 100, c = 'cyan',label = 'Careless') 
        plt.scatter(X[y_kmeans == 4, 0], X[y_kmeans == 4, 1],s = 100, c = 'magenta',label = 'Sensible') 
        plt.scatter(kmeans.cluster_centers_[:,0],kmeans.cluster_centers_[:,1],         ## cluster centers coordinates
                    s = 300, c = 'yellow', label = 'Centroids')
        plt.title('Clusters of clients')
        plt.xlabel('Annual income (k$)')
        plt.ylabel('Spending Score (1-100)')
        plt.legend()
        plt.savefig(img, format='png')
        img.seek(0)
        plot_url = base64.b64encode(img.getvalue())
        
        
    
    return render_template('clustering.html', data = dataset_head.to_html(), describe = describe.to_html(), plot_determine = plot_determine, plot_url=plot_url, rows = rows, columns = columns, clusterers = clusterers, algo = algo)
    

if __name__ == '__main__':
    app.run(host=os.environ.get('IP'),
            port=int(os.environ.get('PORT')),
            debug=True)    
            
  
