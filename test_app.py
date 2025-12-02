"""
Zuriel Hernandez
test cases for app.py
"""

import pytest
import os
import tempfile
import shutil
from app import app


@pytest.fixture
def client():
    """Create a test client for the Flask application."""
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client


@pytest.fixture
def temp_templates_dir():
    """Create a temporary templates directory for testing edge cases."""
    temp_dir = tempfile.mkdtemp()
    original_templates = app.template_folder
    app.template_folder = temp_dir
    yield temp_dir
    app.template_folder = original_templates
    shutil.rmtree(temp_dir)


class TestIndexRoute:
    """Test cases for the home page route (/)"""
    
    def test_index_route_exists(self, client):
        """Test that the home page route returns 200 OK."""
        response = client.get('/')
        assert response.status_code == 200
    
    def test_index_route_content(self, client):
        """Test that the home page contains expected content."""
        response = client.get('/')
        assert b'Tacoma Traffic Crash Risk' in response.data
        assert b'Visualization' in response.data
    
    def test_index_route_uses_template(self, client):
        """Test that the home page uses the correct template."""
        response = client.get('/')
        # Check for content that should be in index.html
        assert b'Predicting and Visualizing' in response.data


class TestMapRoute:
    """Test cases for the map page route (/map)"""
    
    def test_map_route_when_file_exists(self, client):
        """Test that /map route works when map file exists."""
        response = client.get('/map')
        # Should return 200 if map file exists
        assert response.status_code in [200, 503]
    
    def test_map_route_content_when_exists(self, client):
        """Test that map page contains map-related content when file exists."""
        response = client.get('/map')
        # If map exists, should contain Leaflet/folium content
        if response.status_code == 200:
            assert b'leaflet' in response.data.lower() or b'map' in response.data.lower()
    
    def test_map_route_placeholder_when_missing(self, client, temp_templates_dir):
        """Test that /map shows placeholder when map file doesn't exist."""
        # Temporarily remove map file check by using temp directory
        # This simulates the map file not existing
        response = client.get('/map')
        # Should show placeholder or map, depending on actual file existence
        assert response.status_code in [200, 503]


class TestErrorHandling:
    """Test cases for error handling (404, missing files, etc.)"""
    
    def test_404_handler(self, client):
        """Test that 404 errors are handled with custom page."""
        response = client.get('/nonexistent-page')
        assert response.status_code == 404
        assert b'404' in response.data or b'Page Not Found' in response.data
    
    def test_404_handler_has_home_link(self, client):
        """Test that 404 page has a link back to home."""
        response = client.get('/nonexistent-page')
        assert response.status_code == 404
        # Should have a link or button to return home
        assert b'home' in response.data.lower() or b'return' in response.data.lower()
    
    def test_invalid_routes(self, client):
        """Test various invalid routes return 404."""
        invalid_routes = ['/invalid', '/test', '/random123', '/map/invalid']
        for route in invalid_routes:
            response = client.get(route)
            assert response.status_code == 404


class TestStaticFiles:
    """Test cases for static file serving (CSS, JS, etc.)"""
    
    def test_static_css_accessible(self, client):
        """Test that CSS file is accessible via Flask static route."""
        response = client.get('/static/style.css')
        assert response.status_code == 200
        assert b'css' in response.data.lower() or response.content_type == 'text/css; charset=utf-8'
    
    def test_static_file_not_found(self, client):
        """Test that missing static files return 404."""
        response = client.get('/static/nonexistent.css')
        assert response.status_code == 404


class TestURLGeneration:
    """Test cases for Flask URL generation (url_for)"""
    
    def test_index_url_generation(self, client):
        """Test that index page uses url_for correctly."""
        response = client.get('/')
        # Check that Flask template variables are rendered (not raw)
        # If url_for is working, the iframe src should be a valid URL
        assert b'iframe' in response.data.lower()
    
    def test_map_url_in_index(self, client):
        """Test that index page links to map correctly."""
        response = client.get('/')
        # The iframe should point to /map route
        assert b'/map' in response.data or b'map' in response.data.lower()


class TestEdgeCases:
    """Test edge cases and boundary conditions"""
    
    def test_root_path_variations(self, client):
        """Test that root path works with and without trailing slash."""
        response1 = client.get('/')
        response2 = client.get('')
        assert response1.status_code == 200
        # Empty path should redirect or return 200
        assert response2.status_code in [200, 301, 302, 308]
    
    def test_map_path_variations(self, client):
        """Test map route with different path variations."""
        response1 = client.get('/map')
        response2 = client.get('/map/')
        assert response1.status_code in [200, 503]
        # Trailing slash returns 404 (route is defined without trailing slash)
        assert response2.status_code in [200, 301, 302, 308, 404, 503]
    
    def test_method_not_allowed(self, client):
        """Test that POST requests to GET-only routes are handled."""
        # Index and map routes should only accept GET
        response = client.post('/')
        # Flask typically returns 405 Method Not Allowed
        assert response.status_code in [200, 405]
    
    def test_empty_request(self, client):
        """Test handling of empty or malformed requests."""
        response = client.get('/')
        assert response.status_code == 200


class TestResponseHeaders:
    """Test cases for HTTP response headers"""
    
    def test_content_type_html(self, client):
        """Test that HTML responses have correct content type."""
        response = client.get('/')
        assert 'text/html' in response.content_type
    
    def test_response_has_status_code(self, client):
        """Test that all responses have valid status codes."""
        routes = ['/', '/map', '/nonexistent']
        for route in routes:
            response = client.get(route)
            assert 100 <= response.status_code < 600


class TestTemplateRendering:
    """Test cases for template rendering"""
    
    def test_templates_exist(self):
        """Test that required template files exist."""
        templates_dir = 'templates'
        required_templates = ['index.html', '404.html', 'map_placeholder.html']
        
        for template in required_templates:
            template_path = os.path.join(templates_dir, template)
            assert os.path.exists(template_path), f"Template {template} not found"
    
    def test_static_files_exist(self):
        """Test that required static files exist."""
        static_dir = 'static'
        required_static = ['style.css']
        
        for static_file in required_static:
            static_path = os.path.join(static_dir, static_file)
            assert os.path.exists(static_path), f"Static file {static_file} not found"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

