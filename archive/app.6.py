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
    
   
wa.whoosh_index(app, CodeRepo)     
      
@app.route('/')
@app.route('/library')
def library():
    codes = CodeRepo.query.all()
 
    return render_template("library.html", codes = codes)
    
    
@app.route('/search', methods=['GET'])
def search():
    
    codes = CodeRepo.query.whoosh_search(request.args.get('query')).all()
    
    return render_template('library.html', codes = codes)    
    
    
@app.route('/download_code/<code_id>', methods=['GET'])
def download_code(code_id):
    the_code = CodeRepo.query.filter_by(id = code_id).first()
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
    if not 'inputFile' in request.files or not 'name' in request.form:
        return render_template ('bad.html')
    if not 'type_of_algorithm' in request.form or not 'complexity' in request.form:
        return render_template ('bad.html')    
    if not 'method' in request.form or not 'author' in request.form:
        return render_template ('bad.html')    
    
    else:    
        code_file = request.files['inputFile']
        code_file = code_file.read()

        code = CodeRepo(name=request.form['name'],
                        type_of_algorithm=request.form['type_of_algorithm'],
                        complexity=request.form['complexity'],
                        method=request.form['method'],
                        author=request.form['author'],
                        file = code_file)
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
        return render_template('regression.html')
    elif type_id == "Classification": 
        return redirect(url_for("classification"))



@app.route('/classification')
def classification():
    classifiers = Classification.query.all()
    ## Dataset Variables
   
    dataset = pd.read_csv('Social_Network_Ads.csv')
    dataset_head = dataset.head()
    describe = dataset.describe()
    rows = len(dataset.index)
    columns = len(dataset.columns)
    X = dataset.iloc[:, [2, 3]].values
    y = dataset.iloc[:, 4].values  
    pred = '6.5'
    
    return render_template('classification.html', data = dataset_head.to_html(),  describe = describe.to_html(), pred = pred, rows = rows, columns = columns, classifiers = classifiers) 

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
    
    return render_template('classification.html', data = dataset_head.to_html(), describe = describe.to_html(), cma = df.to_html(), plot_url=plot_url, rows = rows, columns = columns, classifiers = classifiers, acc = accuracy)
    
    




if __name__ == '__main__':
    app.run(host=os.environ.get('IP'),
            port=int(os.environ.get('PORT')),
            debug=True)    
            
# class UploadForm(FlaskForm):
#     name = StringField('name', validators=[InputRequired()])
#     type_of_algorithm = StringField('type_of_algorithm', validators=[InputRequired()])
#     complexity = StringField('complexity', validators=[InputRequired()])
#     method = StringField('method', validators=[InputRequired()])
#     author = StringField('author', validators=[InputRequired()])
#     file = FileField('inputfile', validators=[InputRequired()])            
  
