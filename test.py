from app import app
import unittest
from flask_login import current_user

class FlaskTestCase(unittest.TestCase):
    
        def test_index(self):
            tester = app.test_client(self)
            response = tester.get('/dashboard', content_type='html/text')
            self.assertEqual(response.status_code, 200)