import os
from flask import Flask, render_template, redirect, request, url_for, send_file, session
from flask_wtf import FlaskForm
from flask_wtf.file import FileField, FileRequired
from flask_sqlalchemy import SQLAlchemy
from werkzeug.utils import secure_filename
from io import BytesIO

import StringIO
import base64
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt, mpld3
import seaborn as sns

import pandas as pd
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures


app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///database.db'
db = SQLAlchemy(app)






@app.route('/')
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
    # plt.use('Agg')
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