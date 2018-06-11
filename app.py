## App Utilities
import os
from flask import Flask
# , render_template, redirect, request, url_for, send_file, session, request, flash
from flask_sqlalchemy import SQLAlchemy
# from io import BytesIO
# import base64

## Forms

# from flask_wtf import FlaskForm
# from flask_wtf.file import FileField, FileAllowed, FileRequired
# from werkzeug.utils import secure_filename
# from wtforms import StringField, PasswordField, BooleanField
# from wtforms.validators import InputRequired,  DataRequired, Email, Length
from flask_bootstrap import Bootstrap
# from werkzeug.security import generate_password_hash, check_password_hash
from flask_login import LoginManager,login_user, login_required, logout_user, current_user, UserMixin, AnonymousUserMixin


## Data Processing

# import numpy as np
# import matplotlib
# matplotlib.use('Agg')                                                           ## allowing plots to be displayed without opening new window
# import matplotlib.pyplot as plt
# import pandas as pd
# import seaborn as sns


## Machine Learning modules
# from modules import *

## App foundations

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///database.db'
app.config['SECRET_KEY'] = 'BiggestSecret'
db = SQLAlchemy(app)
Bootstrap(app)
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'users.login'
ALLOWED_EXTENSIONS = set(['py'])                                                ## only .py templates allowed for upload


## Import models AFTER db creation (otherwise won't work)

# from models import *
# from forms import *

from project.users.views import users_blueprint
from project.library.views import library_blueprint
from project.algorithms.views import algorithms_blueprint

app.register_blueprint(users_blueprint)
app.register_blueprint(library_blueprint)
app.register_blueprint(algorithms_blueprint)

## Import models AFTER db creation (otherwise won't work)

from project.users.models import User




@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))
    
## Initiate Guest User    
    
class Anonymous(AnonymousUserMixin):
  def __init__(self):
    self.username = 'Guest'
    
    
login_manager.anonymous_user = Anonymous
 

## Database - Users

# class User(UserMixin, db.Model):
#     __tablename__ = 'User'
#     id = db.Column(db.Integer, primary_key=True)
#     username = db.Column(db.String(15), unique=True)
#     email = db.Column(db.String(50), unique=True)
#     password = db.Column(db.String(80))    
    

## Uploaded Templates

# class CodeRepo(db.Model):
#     __searchable__ = ['name', 'type_of_algorithm','complexity','method','author']
#     __tablename__ = 'CodeRepo'
#     id = db.Column(db.Integer, primary_key=True)
#     name = db.Column(db.String(300))
#     type_of_algorithm = db.Column(db.String(300))
#     complexity = db.Column(db.String(300))
#     method = db.Column(db.String(300))
#     author = db.Column(db.String(300))
#     file = db.Column(db.LargeBinary)
#     downloads = db.Column(db.Integer)

## Types of algoritms

# class Types(db.Model):
#     __tablename__ = 'Types'
#     id = db.Column(db.Integer, primary_key=True)
#     type_of_algorithm = db.Column(db.String(300))
    
# ## Algorithm complexity    

# class Complexities(db.Model):
#     __tablename__ = 'Complexity'
#     id = db.Column(db.Integer, primary_key=True)
#     complexity = db.Column(db.String(300))
    
# ## Methods of learning

# class Methods(db.Model):    
#     __tablename__ = 'Methods'
#     id = db.Column(db.Integer, primary_key=True)
#     method = db.Column(db.String(300))
    

# ## Regression Algorithms

# class Regression(db.Model):
#     __tablename__ = 'Regression'
#     id = db.Column(db.Integer, primary_key=True)
#     name = db.Column(db.String(300))
    
# ## Classification Algorithms

# class Classification(db.Model):
#     __tablename__ = 'Classification'
#     id = db.Column(db.Integer, primary_key=True)
#     name = db.Column(db.String(300))
    
# ## Clustering Algorithms    

# class Clustering(db.Model):
#     __tablename__ = 'Clustering'
#     id = db.Column(db.Integer, primary_key=True)
#     name = db.Column(db.String(300))
    
    
# ## Assembler for many-to-many relationship database    
    
# Assembler = db.Table('Assembler',
#     db.Column('algotype_id', db.Integer, db.ForeignKey('AlgoTypes.algotype_id')),
#     db.Column('preprocess_id', db.Integer, db.ForeignKey('Preprocessing.preprocess_id')),
#     db.Column('issue_id', db.Integer, db.ForeignKey('RedFlags.issue_id'))
# )       

# ## Types of algorithms

# class AlgoTypes(db.Model):
#     __tablename__ = 'AlgoTypes'
#     algotype_id = db.Column(db.Integer, primary_key=True)
#     algo_type = db.Column(db.String(300))
#     preprocessers = db.relationship('Preprocessing', secondary=Assembler, 
#                                     backref=db.backref('algos', lazy = 'dynamic'))
#     issues = db.relationship('RedFlags', secondary=Assembler, 
#                                     backref=db.backref('algos', lazy = 'dynamic'))
    

# ## Preprocessing tasks
    
# class Preprocessing(db.Model):
#     __tablename__ = 'Preprocessing'
#     preprocess_id = db.Column(db.Integer, primary_key=True)
#     name = db.Column(db.String(300))    
    
# ##  Potential issues     
    
# class RedFlags(db.Model):
#     __tablename__ = 'RedFlags'
#     issue_id = db.Column(db.Integer, primary_key=True)
#     issue = db.Column(db.String(300))         

# #######################################################################  FORMS & DATABASES ################################################################################################################



# ## Login Form

# class LoginForm(FlaskForm):
#     username = StringField('username', validators=[InputRequired(), Length(min=3, max=15)])
#     password = PasswordField('password', validators=[InputRequired(), Length(min=6, max=80)])
#     remember = BooleanField('Remember Me')

# ## Register Form

# class RegisterForm(FlaskForm):
#     email = StringField('email', validators=[InputRequired(), Email(message='Please enter correct email'), Length(max=50)])
#     username = StringField('username', validators=[InputRequired(), Length(min=3, max=15)])
#     password = PasswordField('password', validators=[InputRequired(), Length(min=6, max=80)])
    

##################################################################### USER LOGIN/SIGN UP TASKS ###########################################################################################################    
    

# ## SIGN UP:

# @app.route('/signup', methods=["GET", "POST"])
# def signup():
#     form = RegisterForm()
#     hashed_password = generate_password_hash(form.password.data, method='sha256')           ## password get hashed for security purposes
#     if form.validate_on_submit(): 
#         new_user = User(username=form.username.data, email=form.email.data, 
#                         password=hashed_password)
#         db.session.add(new_user)
#         db.session.commit()
#         return redirect(url_for('login'))
 
#     return render_template('signup.html', form=form)                            ## passing signup form to signup template
    
    
# ## LOGIN:

# @app.route('/login', methods=["GET", "POST"])
# def login():
#     form = LoginForm()
    
#     if form.validate_on_submit():                                               ## if form was submitted....
#         user = User.query.filter_by(username=form.username.data).first()
#         if user:
#             if check_password_hash(user.password, form.password.data):
#                 login_user(user, remember=form.remember.data)
#                 return redirect(url_for('library'))
        
#         return '<h1>Invalid username or password</h1>'         
    
#     return render_template('login.html', form=form)                             ## passing login form to login template    


# ## LOGOUT

# @app.route('/logout')
# @login_required
# def logout():
#     logout_user()
#     return redirect(url_for('login'))
    
    
    
    
# ########################################################################## MAIN ROUTES ###################################################################################################################    


# ## MAIN DASHBOARD - "LIBRARY"
      
# @app.route('/')
# @app.route('/library')
# def library():
 
#     codes = CodeRepo.query.all()
#     types = Types.query.all()
 
#     return render_template("library.html", codes = codes, types = types, 
#                             user = current_user.username)

# ## DOWNLOAD TEMPLATE    
    
# @app.route('/download_code/<code_id>', methods=['GET'])
# def download_code(code_id):
#     the_code = CodeRepo.query.filter_by(id = code_id).first()
#     click = the_code.downloads + 1                                              ## downloads counter
#     the_code.downloads = click
#     db.session.commit()
    
#     return send_file(BytesIO(the_code.file), 
#                      attachment_filename='{}.py'.format(the_code.name),         ## return file
#                      as_attachment=True)
  
# ## INITIATE TEMPLATE UPLOAD  
    
# @app.route('/add_request')
# def add_request():
#     types = Types.query.all()
#     complexities = Complexities.query.all()
#     methods = Methods.query.all()
    
    
#     return render_template ('add_request.html', types = types, complexities = complexities, 
#                             methods = methods, user = current_user.username)
    
# ## DEFINE ALLOWED TEMPLATE FILE FORMAT    
    
# def allowed_file(filename):
#     return '.' in filename and \
#           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS    
    
    
# ## UPLOAD NEW TEMPLATE HANDLING    
    
# @app.route('/new_code', methods = ['POST'])
# def new_code():
#     codes = CodeRepo.query.all()
    
#     ## Exceptions handling:
    
#     if not 'inputFile' in request.files:
#         return render_template ('error.html', user = current_user.username)
        
#     if not 'type_of_algorithm' in request.form or not 'complexity' in request.form:
#         return render_template ('error.html', user = current_user.username)    
#     if not 'method' in request.form:
#         return render_template ('error.html', user = current_user.username)
#     if request.form['name'] == '' or request.form['author'] == '':
#         return render_template ('error.html', user = current_user.username)
    
#     ## All good
#     else:    
#         code_file = request.files['inputFile']                                  ## check file format    
#         if code_file and allowed_file(code_file.filename):
#             code_file = code_file.read()
#         else:
#             return render_template ('error.html', user = current_user.username) 
            
#         code = CodeRepo(name=request.form['name'],
#                         type_of_algorithm=request.form['type_of_algorithm'],
#                         complexity=request.form['complexity'],
#                         method=request.form['method'],
#                         author=request.form['author'],
#                         file = code_file,
#                         downloads = 0)
#         db.session.add(code)
#         db.session.commit()
#         return redirect(url_for("library"))

# ## EDITING TEMPLATE INITIATION
    
# @app.route('/edit_code/<code_id>')
# def edit_code(code_id):
#     the_code = CodeRepo.query.filter_by(id = code_id).first()
#     types = Types.query.all()
#     complexities = Complexities.query.all()
#     methods = Methods.query.all()
#     return render_template('edit_code.html', code = the_code, types = types, 
#                             complexities = complexities, methods = methods, 
#                             user = current_user.username)    
    
    
# ## UPDATE TEMPLATE HANDLING

# @app.route('/update_code/<code_id>', methods=["POST"])
# def update_code(code_id):
#     the_code = CodeRepo.query.filter_by(id = code_id).first()

#     the_code.name=request.form['name']
#     the_code.type_of_algorithm=request.form['type_of_algorithm']
#     the_code.complexity=request.form['complexity']
#     the_code.method=request.form['method']
#     the_code.author=request.form['author']
    
#     ## Exceptions
    
#     if request.form['name'] == '' or request.form['author'] == '':
#         return render_template ('error.html', user = current_user.username)
#     if 'inputFile' in request.files:
#         code_file = request.files['inputFile']
#         if code_file and allowed_file(code_file.filename):
#             code_file = code_file.read()
#             the_code.file = code_file
#             db.session.commit()
#             return redirect(url_for("library"))
#         else:
#             return render_template ('error.html', user = current_user.username)
            
#     ## All good
    
#     else:    
#         db.session.commit()
#         return redirect(url_for("library"))
    
# ## DELETE TEMPLATE HANDLING    
    
# @app.route('/delete_code/<code_id>')
# def delete_code(code_id):
#     the_code = CodeRepo.query.filter_by(id = code_id).first()
#     db.session.delete(the_code)
#     db.session.commit()
#     return redirect(url_for("library"))
    
    
    
# #################################################################### ML ESTIMATOR TYPES SYMMARY ###########################################################################################################    
    
    
# ## ML TYPE CHOICE HANDLING    
    
# @app.route('/summary/<type_id>')
# def summary(type_id):
#     if type_id == "Regression":
#         return redirect(url_for("regression"))
#     elif type_id == "Classification": 
#         return redirect(url_for("classification"))
#     else:
#         return redirect(url_for("clustering"))
        
# ## REGRESSION INITIATION

# @app.route('/regression')
# def regression():
#     regressors = Regression.query.all()
#     algo = AlgoTypes.query.filter_by(algotype_id = 1).first()
    
#     ## Describe dataset
    
#     dataset = pd.read_csv('Position_Salaries.csv')
#     X = dataset.iloc[:, 1:2].values
#     y = dataset.iloc[:, 2:3].values
#     describe = dataset.describe()
#     rows = len(dataset.index)
#     columns = len(dataset.columns)
#     pred = 'Choose Algorithm'
    
#     return render_template('regression.html', data = dataset.to_html(),  
#                             describe = describe.to_html(), pred = pred, rows = rows, 
#                             columns = columns, regressors = regressors, algo = algo, 
#                             user = current_user.username)
   
# ## REGRESSION ALGORITHM CHOICE HANDLING
    
# @app.route('/regressor/<regressor_id>')
# def regressor(regressor_id):
#     regressors = Regression.query.all() 
#     algo = AlgoTypes.query.filter_by(algotype_id = 1).first()
    
#     ## Describe dataset
#     dataset = pd.read_csv('Position_Salaries.csv')
#     X = dataset.iloc[:, 1:2].values
#     y = dataset.iloc[:, 2:3].values
#     describe = dataset.describe()
#     rows = len(dataset.index)
#     columns = len(dataset.columns)
#     pred = 'Choose Algorithm'
#     choice = regressor_id
#     plt.gcf().clear()
    
#     if choice == '1':
#         ## Fitting Polynomial Regression to the dataset
#         poly_reg = PolynomialFeatures(degree = 4)                               ## Transform X into new matrix composed with X^4
#         X_poly = poly_reg.fit_transform(X)                                      ## First fit, then transform the object into poly - used in a plot later        
#         lin_reg = LinearRegression()
#         lin_reg.fit(X_poly, y)
        
#         ## Visualising The Polynomial Regression
#         img = BytesIO()                                               ## initiate an image for holding the plot
#         plt.gcf().clear()                                                       ## clearing previous plots from canvas
#         X_grid = np.arange(min(X), max(X), 0.1)   
#         X_grid = X_grid.reshape(len(X_grid),1)  
#         sns.set_style("darkgrid", {"axes.facecolor": ".9"})
#         plt.scatter(X,y, color = 'red')
#         plt.plot(X_grid, lin_reg.predict(poly_reg.fit_transform(X_grid)),color ='darkblue')    
#         plt.ylim(ymin=0)
#         plt.title('Salary Estimate - Polynomial Regression')
#         plt.xlabel('Position Level')
#         plt.ylabel('Salary', fontsize=12)
#         plt.yticks(fontsize=10)
        
#         plt.savefig(img, format='png')                                          ## save generated plot as an image    
#         img.seek(0)
#         plot_url = base64.b64encode(img.getvalue())
#         plt.gcf().clear()
        
#         ## Predicting result with Polynomial Regression
#         pred = lin_reg.predict(poly_reg.fit_transform(6.5))
#         pred = pred.flat[0]
#         pred = ('%.2f' % (pred,)).rstrip('0').rstrip('.')
        
#     elif choice == '2':
#         # Feature Scaling - needed for rbf SVR
#         sc_X = StandardScaler()
#         X = sc_X.fit_transform(X)
#         sc_y = StandardScaler()
#         y = sc_y.fit_transform(y)
        
#         # Fitting SVR to the dataset
#         regressor = SVR(kernel = 'rbf')                                         ## Kernel choice - linear, poly or gaussian(rbf)
#         regressor.fit(X,y)
        
#         img = BytesIO()
#         X_grid = np.arange(min(X), max(X), 0.1)
#         X_grid = X_grid.reshape((len(X_grid), 1))
#         plt.scatter(X, y, color = 'red')
#         sns.set_style("darkgrid", {"axes.facecolor": ".9"})
#         plt.plot(X_grid, regressor.predict(X_grid), color = 'darkblue')
#         plt.title('Salary Estimate - SVR')
#         plt.xlabel('Position level')
#         plt.ylabel('Salary', fontsize=12)
#         plt.yticks(fontsize=10)
#         plt.savefig(img, format='png')
#         img.seek(0)
#         plot_url = base64.b64encode(img.getvalue())
#         plt.gcf().clear()
        
#         # Predicting result - has to be changed as scaling was applied 
#         pred = sc_y.inverse_transform(regressor.predict(sc_X.transform(np.array([[6.5]]))))     ## transform value into array with numpy.array
#         pred = pred.flat[0]
#         pred = ('%.2f' % (pred,)).rstrip('0').rstrip('.')                       ## for better display
#     else: 
        
#         # Fitting Random Forest Regression Model to the dataset
#         regressor = RandomForestRegressor(n_estimators = 600, random_state = 0)
#         regressor.fit(X,y)
        
#         # Visualising the Regression results (for higher resolution and smoother curve)
#         img = BytesIO()
#         X_grid = np.arange(min(X), max(X), 0.01)                                ## increase the resolution
#         X_grid = X_grid.reshape((len(X_grid), 1))   
#         plt.scatter(X, y, color = 'red')
#         sns.set_style("darkgrid", {"axes.facecolor": ".9"})
#         plt.plot(X_grid, regressor.predict(X_grid), color = 'darkblue')
#         plt.ylim(ymin=0)
#         plt.title('Salary Estimate - RandomForest Regression')
#         plt.xlabel('Position level')
#         plt.ylabel('Salary', fontsize=12)
#         plt.yticks(fontsize=10)
#         plt.savefig(img, format='png')
#         img.seek(0)
#         plot_url = base64.b64encode(img.getvalue())
#         plt.gcf().clear()
        
#         # Predicting result
#         pred = regressor.predict(6.5)
#         pred = pred.flat[0]
#         pred = ('%.2f' % (pred,)).rstrip('0').rstrip('.')
        
#     return render_template('regression.html', data = dataset.to_html(), 
#                             describe = describe.to_html(), pred = pred, 
#                             plot_url=plot_url, rows = rows, columns = columns, 
#                             regressors = regressors, algo = algo, user = current_user.username)

        
# ## CLASSIFICATION INITIATION

# @app.route('/classification')
# def classification():
#     classifiers = Classification.query.all()
#     algo = AlgoTypes.query.filter_by(algotype_id = 2).first()
    
#     ## Describe dataset
   
#     dataset = pd.read_csv('Social_Network_Ads.csv')
#     dataset_head = dataset.head(10)
#     stats_data = dataset.iloc[:,2:4]
#     describe = stats_data.describe()
#     rows = len(dataset.index)
#     columns = len(dataset.columns)
#     X = dataset.iloc[:, [2, 3]].values
#     y = dataset.iloc[:, 4].values  
#     pred = 'Choose Algorithm'
    
#     return render_template('classification.html', data = dataset_head.to_html(),  
#                             describe = describe.to_html(), pred = pred, rows = rows, 
#                             columns = columns, classifiers = classifiers, 
#                             algo = algo, user = current_user.username) 

# ## CLASSIFICATION ALGORITHM CHOICE HANDLING

# @app.route('/classifier/<classifier_id>')
# def classifier(classifier_id):
#     classifiers = Classification.query.all()
#     algo = AlgoTypes.query.filter_by(algotype_id = 2).first()
    
#     ## Describe dataset
#     dataset = pd.read_csv('Social_Network_Ads.csv')
#     dataset_head = dataset.head(10)
#     stats_data = dataset.iloc[:,2:4]
#     describe = stats_data.describe()
#     rows = len(dataset.index)
#     columns = len(dataset.columns)
#     X = dataset.iloc[:, [2, 3]].values
#     y = dataset.iloc[:, 4].values  
#     plt.gcf().clear()
    
#     # Splitting the dataset into the Training set and Test set
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)
    
#     # Feature Scaling
#     sc = StandardScaler()
#     X_train = sc.fit_transform(X_train)
#     X_test = sc.transform(X_test)
    
#     # Choose classifier to fit the Training set
#     classifier = GaussianNB() if classifier_id == '3'  \
#                               else KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2) if classifier_id == '1' \
#                               else SVC(kernel = 'rbf', random_state = 0 )
#     classifier.fit(X_train, y_train)
    
#     # Predicting the Test set results
#     y_pred = classifier.predict(X_test)
    
#     # Making the Confusion Matrix
#     from sklearn.metrics import confusion_matrix
#     cm = confusion_matrix(y_test, y_pred)
#     df = pd.DataFrame(cm, index = ['Bought: Yes', 'Bought: No'], columns = ['Predicted: Yes', 'Predicted: No'])
#     accuracy = (df.iloc[0,0] + df.iloc[1,1])
#     accuracy =  "{}%".format(float(accuracy))
    
    
#     # Visualising the Test set results with the heatmap
#     from matplotlib.colors import ListedColormap
#     img = BytesIO()
    
#     X_set, y_set = X_test, y_test
#     X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
#                          np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
#     plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
#                  alpha = 0.85, cmap = ListedColormap(('darkred', 'darkgreen')))
#     plt.xlim(X1.min(), X1.max())
#     plt.ylim(X2.min(), X2.max())
#     for i, j in enumerate(np.unique(y_set)):
#         plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
#                     c = ListedColormap(('red', 'green'))(i), label = j)
#     plt.title('Test Set Results')
#     plt.xlabel('Age')
#     plt.ylabel('Estimated Salary')
#     plt.legend()
#     plt.savefig(img, format='png')
#     img.seek(0)
#     plot_url = base64.b64encode(img.getvalue())
#     plt.gcf().clear()
    
    
    
#     return render_template('classification.html', data = dataset_head.to_html(), 
#                             describe = describe.to_html(), cma = df.to_html(), 
#                             plot_url=plot_url, rows = rows, columns = columns, 
#                             classifiers = classifiers, acc = accuracy, 
#                             algo = algo, user = current_user.username)
    

# ## CLUSTERING INITIATION

# @app.route('/clustering')
# def clustering():
#     clusterers = Clustering.query.all()
#     algo = AlgoTypes.query.filter_by(algotype_id = 3).first()
    
#     ## Describe dataset
#     dataset = pd.read_csv('Mall_Customers.csv')
#     X = dataset.iloc[:, [3, 4]].values
#     dataset_head = dataset.head(10)
#     stats_data = dataset.iloc[:,2:5]
#     describe = stats_data.describe()
#     rows = len(dataset.index)
#     columns = len(dataset.columns)
#     pred = 'Choose Algorithm'
    
#     return render_template('clustering.html', data = dataset_head.to_html(),  
#                             describe = describe.to_html(), pred = pred, 
#                             rows = rows, columns = columns, 
#                             clusterers = clusterers, algo = algo, 
#                             user = current_user.username)
   
# ## CLUSTERING ALGORITHM CHOICE HANDLING   
    
# @app.route('/clusterer/<clusterer_id>')
# def clusterer(clusterer_id):
#     clusterers = Clustering.query.all()
#     algo = AlgoTypes.query.filter_by(algotype_id = 3).first()
    
#     ## Describe dataset
#     dataset = pd.read_csv('Mall_Customers.csv')
#     X = dataset.iloc[:, [3, 4]].values
#     dataset_head = dataset.head(10)
#     stats_data = dataset.iloc[:,2:5]
#     describe = stats_data.describe()
#     rows = len(dataset.index)
#     columns = len(dataset.columns)
#     pred = 'Choose Algorithm'
    
#     choice = clusterer_id
    
#     if choice == '2':
        
#         ## Using the dendogram to find the optimal number of clusters
#         plt.gcf().clear()
#         dendrogram = sch.dendrogram(sch.linkage(X,method = 'ward'))
#         img_dendrogram = BytesIO()
#         sns.set_style("darkgrid", {"axes.facecolor": ".9"})
#         plt.title("Dendrogram")
#         plt.xlabel('Customers')
#         plt.ylabel('Euclidean distances')
#         plt.savefig(img_dendrogram, format='png')
#         img_dendrogram.seek(0)
#         plot_determine = base64.b64encode(img_dendrogram.getvalue())
#         ## Result = 5
        
#         ## Fitting Hierarchical Clustering to the dataset (optimal clusters = 5)
#         hc = AgglomerativeClustering(n_clusters = 5, affinity = 'euclidean', linkage = 'ward')
#         y_hc = hc.fit_predict(X)
        
#         ## Visualising the clusters 
#         img = BytesIO()
#         plt.gcf().clear()
#         plt.scatter(X[y_hc == 0, 0], X[y_hc == 0, 1],   ## specify that we want first cluster + first column vs second column for 'y'
#             s = 100, c = 'red',label = 'Savers')                            ## size for datapoints/color
#         plt.scatter(X[y_hc == 1, 0], X[y_hc == 1, 1],s = 100, c = 'blue',label = 'Average') 
#         plt.scatter(X[y_hc == 2, 0], X[y_hc == 2, 1],s = 100, c = 'green',label = 'Target Group') 
#         plt.scatter(X[y_hc == 3, 0], X[y_hc == 3, 1],s = 100, c = 'orange',label = 'Overspenders') 
#         plt.scatter(X[y_hc == 4, 0], X[y_hc == 4, 1],s = 100, c = 'magenta',label = 'Careful') 
#         sns.set_style("darkgrid", {"axes.facecolor": ".9"})
#         plt.title('Suggested Clusters')
#         plt.xlabel('Annual income (k$)')
#         plt.ylabel('Spending Score (1-100)', fontsize=12)
#         plt.ylim(ymin=0)
#         plt.legend(fontsize = 9)
#         plt.savefig(img, format='png')
#         img.seek(0)
#         plot_url = base64.b64encode(img.getvalue())
        
#     if choice == '1':    
        
#         ## Using the elbow method to find the optimal number of clusters
#         wcss = []    ## initialize the list
#         for i in range(1, 11):
#             kmeans = KMeans(n_clusters = i,        ## from 1 to 10
#                             init = 'k-means++',    ## k-means++ to avoid random initialziation trap
#                             max_iter = 300,        ## 300 is deafault        
#                             n_init  = 10,          ## algorithm runs with different initial centroids      
#                             random_state = 0)
#             kmeans.fit(X)
#             wcss.append(kmeans.inertia_)           ## to compute wcss   
#         ## Result = 5    
        
#         ## Visualising Elbow Method
#         plt.gcf().clear()
#         img_elbow = BytesIO()
#         plt.plot(range(1,11), wcss)
#         plt.title('The Elbow Method')
#         plt.xlabel('Number of clusters')
#         plt.ylabel('WCSS')
#         sns.set_style("darkgrid", {"axes.facecolor": ".9"})
#         plt.savefig(img_elbow, format='png')
#         img_elbow.seek(0)
#         plot_determine = base64.b64encode(img_elbow.getvalue())
        
        
        
#         ## Applying k-means to the mall dataset - from the plot we can see that optimum is 5 clusters.
#         kmeans = KMeans(n_clusters = 5,
#                         init = 'k-means++',    ## k-means++ to avoid random initialziation trap
#                         max_iter = 300,        ## 300 is deafault        
#                         n_init  = 10,          ## algorithm runs with different initial centroids      
#                         random_state = 0)
#         y_kmeans = kmeans.fit_predict(X)       ## fit_predict returns a cluster for each observation 
        
#         ## Visualising the clusters
#         img = BytesIO()
#         plt.gcf().clear()
#         sns.set_style("darkgrid", {"axes.facecolor": ".9"})
#         plt.scatter(X[y_kmeans == 0, 0], X[y_kmeans == 0, 1],   ## specify that we want first cluster + first column vs second column for 'y'
#                     s = 100, c = 'red',label = 'Savers')                            ## size for datapoints/color
#         plt.scatter(X[y_kmeans == 1, 0], X[y_kmeans == 1, 1],s = 100, c = 'blue',label = 'Average') 
#         plt.scatter(X[y_kmeans == 2, 0], X[y_kmeans == 2, 1],s = 100, c = 'green',label = 'Target Group') 
#         plt.scatter(X[y_kmeans == 3, 0], X[y_kmeans == 3, 1],s = 100, c = 'orange',label = 'Overspenders') 
#         plt.scatter(X[y_kmeans == 4, 0], X[y_kmeans == 4, 1],s = 100, c = 'magenta',label = 'Careful') 
#         plt.scatter(kmeans.cluster_centers_[:,0],kmeans.cluster_centers_[:,1],         ## cluster centers coordinates
#                     s = 200, c = 'black', label = 'Centroids')
#         plt.title('Suggested Clusters')
#         plt.xlabel('Annual income (k$)')
#         plt.ylabel('Spending Score (1-100)', fontsize=12)
#         plt.ylim(ymin=0)
#         plt.legend(fontsize = 9)
#         plt.savefig(img, format='png')
#         img.seek(0)
#         plot_url = base64.b64encode(img.getvalue())
        
        
#     return render_template('clustering.html', data = dataset_head.to_html(), 
#                             describe = describe.to_html(), plot_determine = plot_determine, 
#                             plot_url=plot_url, rows = rows, columns = columns, 
#                             clusterers = clusterers, algo = algo, 
#                             user = current_user.username)
    
    
## APP INITIATION

if __name__ == '__main__':
    app.run(host=os.environ.get('IP'),
            port=int(os.environ.get('PORT')),
            debug=True)    
            
  
