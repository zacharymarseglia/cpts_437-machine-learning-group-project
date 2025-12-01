"""
Flask application for Tacoma Traffic Crash Risk Predictor
Serves the landing page and interactive crash risk map visualization.
Author: Zuriel H
"""

from flask import Flask, render_template, url_for
import os

app = Flask(__name__)

# Configuration
TEMPLATES_DIR = 'templates'
STATIC_DIR = 'static'
MAP_FILE = 'tacoma_crash_risk_map.html'


@app.route('/')
def index():
    """
    Render the landing page (index.html).
    
    Returns:
        Rendered HTML template for the landing page.
    """
    return render_template('index.html')


@app.route('/map')
def map_page():
    """
    Render the interactive crash risk map.
    
    Handles the case where the map file doesn't exist yet (e.g., ML model hasn't run).
    In such cases, renders a friendly placeholder page instead of crashing.
    
    Returns:
        Rendered HTML template for the map, or a placeholder if map doesn't exist.
    """
    map_path = os.path.join(TEMPLATES_DIR, MAP_FILE)
    
    # Check if map file exists
    if os.path.exists(map_path):
        return render_template(MAP_FILE)
    else:
        return render_template('map_placeholder.html'), 503


@app.errorhandler(404)
def page_not_found(error):
    """
    Handle 404 errors with a custom error page.
    
    Args:
        error: The error object from Flask.
        
    Returns:
        Rendered HTML template for 404 error page.
    """
    return render_template('404.html'), 404


if __name__ == '__main__':
    # Run the Flask development server
    # Set debug=True for development, False for production
    port = int(os.environ.get('PORT', 5000))
    debug = os.environ.get('FLASK_ENV') != 'production'
    app.run(debug=debug, host='0.0.0.0', port=port)

