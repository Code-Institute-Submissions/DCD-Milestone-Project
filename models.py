# from app import db
# from flask_login import UserMixin

# ## Database - Users

# class User(UserMixin, db.Model):
#     __tablename__ = 'User'
#     id = db.Column(db.Integer, primary_key=True)
#     username = db.Column(db.String(15), unique=True)
#     email = db.Column(db.String(50), unique=True)
#     password = db.Column(db.String(80))    
    

# ## Uploaded Templates

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

# ## Types of algoritms

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