from app import app, User
import unittest
from flask_login import current_user

class UsersTestCase(unittest.TestCase):
    
        def test_login(self):
            tester = app.test_client(self)
            response = tester.get('/login', content_type='html/text')
            self.assertEqual(response.status_code, 200)
            
        # Login page loads correctly  
        def test_login_page_loads(self):
            tester = app.test_client(self)
            response = tester.get('/login', content_type='html/text')
            self.assertEqual(response.status_code, 200)
            self.assertTrue(b'Please Log in' in response.data)     
            
        def test_correct_signup(self):
            tester = app.test_client(self)
            response = tester.post('/signup', data=dict(username="admin", 
                                                        email="admin@gmail.com", 
                                                        password="adminadmin"), 
                                                        follow_redirects = True)
            self.assertTrue(b'sign' in response.data)         
            
        # Correct login credentials    
        def test_correct_login(self):
            tester = app.test_client(self)
            response = tester.post('/login', data=dict(username="admin", password="adminadmin"), 
                                                follow_redirects = True)
            self.assertTrue(b'Please Log in' in response.data)     

        # Login works correctly given incorrect credentials    
        def test_incorrect_login(self):
            tester = app.test_client(self)
            response = tester.post('/login', data=dict(username="admin", password="zzz"), 
                                                follow_redirects = True)
            self.assertTrue(b'Field must be between 6' in response.data)               
            
        # Ensure that main page @logout_required    
        def test_logout_requires_login(self):
            tester = app.test_client(self)
            response = tester.get('/logout', follow_redirects = True)
            self.assertTrue(b'Please Log in' in response.data)    
            
            
class AlgorithmsTestCase(unittest.TestCase):

            

        # Classification Page Test
        def test_classification_page_loads(self):
            tester = app.test_client(self)
            response = tester.get('/classification', content_type='html/text')
            self.assertEqual(response.status_code, 200)
            self.assertTrue(b'classification' in response.data) 
            
        # Regression Page Test    
        def test_regression_page_loads(self):
            tester = app.test_client(self)
            response = tester.get('/regression', content_type='html/text')
            self.assertEqual(response.status_code, 200)
            self.assertTrue(b'regression' in response.data) 
            
        # Clustering Page Test    
        def test_clustering_page_loads(self):
            tester = app.test_client(self)
            response = tester.get('/clustering', content_type='html/text')
            self.assertEqual(response.status_code, 200)
            self.assertTrue(b'clustering' in response.data) 



class LibraryTestCase(unittest.TestCase):

            
        # Classification Page Test
        def test_add_request_page_loads(self):
            tester = app.test_client(self)
            response = tester.get('/add_request', content_type='html/text')
            self.assertEqual(response.status_code, 200)
            self.assertTrue(b'Add Template' in response.data) 
            
        # Regression Page Test    
        def test_edit_code_page_loads(self):
            tester = app.test_client(self)
            response = tester.get('/edit_code/1', content_type='html/text')
            self.assertEqual(response.status_code, 200)
            self.assertTrue(b'Edit Template' in response.data) 
            
        # Clustering Page Test    
        def test_library_page_loads(self):
            tester = app.test_client(self)
            response = tester.get('/', content_type='html/text')
            self.assertEqual(response.status_code, 200)
            self.assertTrue(b'Available Code Templates' in response.data) 
     
            

if __name__ == '__main__':
    unittest.main()            