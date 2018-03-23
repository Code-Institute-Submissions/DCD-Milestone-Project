import os
from flask import Flask, render_template, redirect, request, url_for, send_file
from flask_wtf import FlaskForm
from flask_wtf.file import FileField, FileRequired
from flask_sqlalchemy import SQLAlchemy
from werkzeug.utils import secure_filename
from io import BytesIO


app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///database.db'
db = SQLAlchemy(app)


textcode = open('code.txt', 'r')


class CodeRepo(db.Model):
    __tablename__ = 'CodeRepo'
    id = db.Column(db.Integer, primary_key=True)
    model = db.Column(db.String(300))
    type_of_algorithm = db.Column(db.String(300))
    complexity = db.Column(db.String(300))
    learning_method = db.Column(db.String(300))
    preprocessing = db.Column(db.String(300))
    
    def __init__(self, id, model, type_of_algorithm, complexity,learning_method, preprocessing):
        self.id = id
        self.model = model
        self.type_of_algorithm = type_of_algorithm
        self.complexity = complexity
        self.learning_method = learning_method
        self.preprocessing = preprocessing
                                    
    
@app.route('/')
@app.route('/library')
def library():
    codes = CodeRepo.query.all()
 
    return render_template("library.html", codes = codes)
    
    
@app.route('/add_request')
def add_request():
    return render_template ('add_request.html')
    
    
@app.route('/add_code', methods = ['POST'])
def add_code():
    code = CodeRepo(id = request.form['identity'],
                    model=request.form['model'],
                    type_of_algorithm=request.form['type_of_algorithm'],
                    complexity=request.form['complexity'],
                    learning_method=request.form['learning_method'],
                    preprocessing=request.form['preprocessing'])
    db.session.add(code)
    db.session.commit()
    
    return render_template("library.html", code = code)
    
    
    
if __name__ == '__main__':
    app.run(host=os.environ.get('IP'),
            port=int(os.environ.get('PORT')),
            debug=True)    