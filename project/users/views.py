from flask import Flask, render_template, redirect, request, url_for, send_file, session, request, flash, Blueprint
from flask_login import LoginManager,login_user, login_required, logout_user, current_user, UserMixin, AnonymousUserMixin


## Blueprint Init

users_blueprint = Blueprint(
    'users', __name__,
    template_folder="templates"
    )

from app import db
from forms import *
from models import User

########################################################################## USER VIEWS ###################################################################################################################  


## SIGN UP ##########################################################################

@users_blueprint.route('/signup', methods=["GET", "POST"])
def signup():
    form = RegisterForm()
    hashed_password = generate_password_hash(form.password.data, method='sha256')           ## password get hashed for security purposes
    if form.validate_on_submit(): 
        new_user = User(username=form.username.data, email=form.email.data, 
                        password=hashed_password)
        db.session.add(new_user)
        db.session.commit()
        return redirect(url_for('users.login'))
 
    return render_template('signup.html', form=form)                            ## passing signup form to signup template
    
    
## LOGIN ############################################################################

@users_blueprint.route('/login', methods=["GET", "POST"])
def login():
    form = LoginForm()
    message = '<h1>Invalid username or password</h1>'  
    
    if form.validate_on_submit():                                               ## if form was submitted....
        user = User.query.filter_by(username=form.username.data).first()
        if user:
            if check_password_hash(user.password, form.password.data):
                login_user(user, remember=form.remember.data)
                return redirect(url_for('library.library'))
        
        return render_template('login.html', form=form, message=message)          
    
    return render_template('login.html', form=form)                             ## passing login form to login template    


## LOGOUT ###########################################################################

@users_blueprint.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('users.login'))