from app import db


## Regression Algorithms

class Regression(db.Model):
    __tablename__ = 'Regression'
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(300))
    
## Classification Algorithms

class Classification(db.Model):
    __tablename__ = 'Classification'
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(300))
    
## Clustering Algorithms    

class Clustering(db.Model):
    __tablename__ = 'Clustering'
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(300))
    
    
## Assembler for many-to-many relationship database    
    
Assembler = db.Table('Assembler',
    db.Column('algotype_id', db.Integer, db.ForeignKey('AlgoTypes.algotype_id')),
    db.Column('preprocess_id', db.Integer, db.ForeignKey('Preprocessing.preprocess_id')),
    db.Column('issue_id', db.Integer, db.ForeignKey('RedFlags.issue_id'))
)       

## Types of algorithms

class AlgoTypes(db.Model):
    __tablename__ = 'AlgoTypes'
    algotype_id = db.Column(db.Integer, primary_key=True)
    algo_type = db.Column(db.String(300))
    preprocessers = db.relationship('Preprocessing', secondary=Assembler, 
                                    backref=db.backref('algos', lazy = 'dynamic'))
    issues = db.relationship('RedFlags', secondary=Assembler, 
                                    backref=db.backref('algos', lazy = 'dynamic'))
    

## Preprocessing tasks
    
class Preprocessing(db.Model):
    __tablename__ = 'Preprocessing'
    preprocess_id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(300))    
    
##  Potential issues     
    
class RedFlags(db.Model):
    __tablename__ = 'RedFlags'
    issue_id = db.Column(db.Integer, primary_key=True)
    issue = db.Column(db.String(300))         