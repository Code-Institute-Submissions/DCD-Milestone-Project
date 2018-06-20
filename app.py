## App Utilities
import os
from flask import Flask, render_template
from flask_sqlalchemy import SQLAlchemy
from flask_bootstrap import Bootstrap
from flask_login import LoginManager,login_user, login_required, logout_user, current_user, UserMixin, AnonymousUserMixin




## App Settings

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///database.db'
app.config['SECRET_KEY'] = 'BiggestSecret'
db = SQLAlchemy(app)
Bootstrap(app)
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'users.login'
ALLOWED_EXTENSIONS = set(['py'])                                                ## only .py templates allowed for upload



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


@app.errorhandler(404)
def error404(error):
    return render_template('404.html'), 404
    
@app.errorhandler(500)
def error500(error):
    return render_template('500.html'), 500
 


## APP INITIATION

if __name__ == '__main__':
    # Bind to PORT if defined, otherwise default to 5000.
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False) 
            
