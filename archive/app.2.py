## APP
import os
from flask import Flask, render_template, redirect, request, url_for, send_file, session
from flask_wtf import FlaskForm
from flask_wtf.file import FileField, FileRequired
from flask_sqlalchemy import SQLAlchemy
import flask_whooshalchemy as wa
from werkzeug.utils import secure_filename
from io import BytesIO
import StringIO
import base64

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


app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///database.db'
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
    
    # def __init__(self, id, name, type_of_algorithm, complexity, method, author):
    #     self.id = id
    #     self.name = name
    #     self.type_of_algorithm = type_of_algorithm
    #     self.complexity = complexity
    #     self.method = method
    #     self.author = author

class Types(db.Model):
    __tablename__ = 'Types'
    id = db.Column(db.Integer, primary_key=True)
    type_of_algorithm = db.Column(db.String(300))
    
    # def __init__(self, id, type_of_algorithm):
    #     self.id = id
    #     self.type_of_algorithm = type_of_algorithm
        
class Complexities(db.Model):
    __tablename__ = 'Complexity'
    id = db.Column(db.Integer, primary_key=True)
    complexity = db.Column(db.String(300))
    
    # def __init__(self, id, complexity):
    #     self.id = id
    #     self.complexity = complexity
        
class Methods(db.Model):    
    __tablename__ = 'Methods'
    id = db.Column(db.Integer, primary_key=True)
    method = db.Column(db.String(300))
    
    # def __init__(self, id, method):
    #     self.id = id
    #     self.method = method
    
     
      
@app.route('/')
@app.route('/library')
def library():
    codes = CodeRepo.query.all()
 
    return render_template("library.html", codes = codes)
    
    
@app.route('/download_code/<code_id>', methods=['GET'])
def download_code(code_id):
    the_code = CodeRepo.query.filter_by(id = code_id).first()
    return send_file(BytesIO(the_code.file), attachment_filename='{}.pdf'.format(the_code.name), as_attachment=True)      
    
    
@app.route('/add_request')
def add_request():
    types = Types.query.all()
    complexities = Complexities.query.all()
    methods = Methods.query.all()
    
    
    return render_template ('add_request.html', types = types, complexities = complexities, methods = methods )
    
    
@app.route('/new_code', methods = ['POST'])
def new_code():
    codes = CodeRepo.query.all()
    ## fix author
    if not 'inputFile' in request.files:
        return render_template ('bad.html')
    elif not 'type_of_algorithm'in request.form:
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
    # if not 'name'in request.form or not 'author' in request.form:
    #     return render_template ('bad.html')
    # else:
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
    if type_id == "regression":
        return render_template('regression.html')
    elif type_id == "classification": 
        return redirect(url_for("classification"))






@app.route('/classification')
def classification():
    dataset = pd.read_csv('data.csv')
    X = dataset.iloc[:,1:2].values
    y = dataset.iloc[:, 2].values
    
    poly_reg = PolynomialFeatures(degree = 4)          
    X_poly = poly_reg.fit_transform(X)                     
    lin_reg = LinearRegression()
    lin_reg.fit(X_poly, y)
    
    img = StringIO.StringIO()
    sns.set_style("darkgrid")
    sns.set_context("paper")
    X_grid = np.arange(min(X), max(X), 0.1)   
    X_grid = X_grid.reshape(len(X_grid),1)   
    plt.scatter(X,y, color = 'black')
    plt.plot(X_grid, lin_reg.predict(poly_reg.fit_transform(X_grid)), color = 'teal')    
    plt.title('Reality Check (Polynomial Regression)')
    plt.xlabel('Position Level')
    plt.ylabel('Salary')
    # plt.show() 
    plt.savefig(img, format='png')
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue())
    
    pred = lin_reg.predict(poly_reg.fit_transform(6.5))
    
    return render_template('classification.html', data = X, pred = pred, plot_url=plot_url) 



if __name__ == '__main__':
    app.run(host=os.environ.get('IP'),
            port=int(os.environ.get('PORT')),
            debug=True)    