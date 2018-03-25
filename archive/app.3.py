import os
from flask import Flask, render_template, redirect, request, url_for, send_file, session
from flask_wtf import FlaskForm
from flask_wtf.file import FileField, FileRequired
from flask_sqlalchemy import SQLAlchemy
from werkzeug.utils import secure_filename
from io import BytesIO


app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///database.db'
db = SQLAlchemy(app)


class CodeRepo(db.Model):
    __tablename__ = 'CodeRepo'
    id = db.Column(db.Integer, primary_key=True)
    model = db.Column(db.String(300))
    type_of_algorithm = db.Column(db.String(300))
    complexity = db.Column(db.String(300))
    learning_method = db.Column(db.String(300))
    preprocessing = db.Column(db.String(300))
    
    
class Files(db.Model):
    __tablename__ = 'files'
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(300))
    data = db.Column(db.LargeBinary) 
    
    
class Categories(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    type_of_algorithm = db.Column(db.String(300))
    def __init__(self, id, type_of_algorithm):
        self.id = id
        self.type_of_algorithm = type_of_algorithm
      


@app.route('/')
@app.route('/library')
def library():
    codes = CodeRepo.query.all()
 
    return render_template("library.html", codes = codes)
    
    
@app.route('/add_request')
def add_request():
    return render_template ('add_request.html')
    
    
@app.route('/new_code', methods = ['POST'])
def new_code():
    codes = CodeRepo.query.all()
    code = CodeRepo(model=request.form['model'],
                    type_of_algorithm=request.form['type_of_algorithm'],
                    complexity=request.form['complexity'],
                    learning_method=request.form['learning_method'],
                    preprocessing=request.form['preprocessing'])
    db.session.add(code)
    db.session.commit()
    
    return render_template("library.html", codes = codes)
    
    
@app.route('/edit_code/<code_id>')
def edit_code(code_id):
    the_code = CodeRepo.query.filter_by(id = code_id).first()
    return render_template('edit_code.html', code = the_code)    
    
    
@app.route('/update_code/<code_id>', methods=["POST"])
def update_code(code_id):
    the_code = CodeRepo.query.filter_by(id = code_id).first()
    the_code.model=request.form['model']
    the_code.type_of_algorithm=request.form['type_of_algorithm']
    the_code.complexity=request.form['complexity']
    the_code.learning_method=request.form['learning_method']
    the_code.preprocessing=request.form['preprocessing']
    db.session.commit()
    return redirect(url_for("library"))
    
    
@app.route('/delete_code/<code_id>')
def delete_code(code_id):
    the_code = CodeRepo.query.filter_by(id = code_id).first()
    db.session.delete(the_code)
    db.session.commit()
    return redirect(url_for("library"))
    
@app.route('/upload', methods=['POST'])
def upload():
    file = request.files['inputFile']
    
    newFile = FileContents(name=file.filename, data=file.read())
    db.session.add(newFile)
    db.session.commit()    
    return render_template('edit_code.html')
    
@app.route('/download', methods=['GET'])
def download():
    file_data = FileContents.query.filter_by(id=1).first()
    return send_file(BytesIO(file_data.data), attachment_filename='file.pdf', as_attachment=True)    
    
    
    
if __name__ == '__main__':
    app.run(host=os.environ.get('IP'),
            port=int(os.environ.get('PORT')),
            debug=True)    