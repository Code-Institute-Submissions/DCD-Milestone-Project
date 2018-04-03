assembler = db.Table('assembler',
    db.Column('algotype_id', db.Integer, db.ForeignKey('AlgoTypes.algotype_id')),
    db.Column('preprocess_id', db.Integer, db.ForeignKey('Preprocessing.preprocess_id')),
    db.Column('issue_id', db.Integer, db.ForeignKey('RedFlags.issue_id'))
)       

class AlgoTypes(db.Model):
    __tablename__ = 'AlgoTypes'
    algotype_id = db.Column(db.Integer, primary_key=True)
    algo_type = db.Column(db.String(300))
    preprocessers = db.relationship('Preprocessing', secondary=assembler, backref=db.backref('algos', lazy = 'dynamic'))
    issues = db.relationship('RedFlags', secondary=assembler, backref=db.backref('algos', lazy = 'dynamic'))
    
    
class Preprocessing(db.Model):
    __tablename__ = 'Preprocessing'
    preprocess_id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(300))    
    
class RedFlags(db.Model):
    __tablename__ = 'RedFlags'
    issue_id = db.Column(db.Integer, primary_key=True)
    issue = db.Column(db.String(300))       