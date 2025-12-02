"""
Flask application for Tacoma Traffic Crash Risk Predictor
Serves the landing page and interactive crash risk map visualization.
Author: Zuriel H
"""

from flask import Flask, render_template, url_for, request, jsonify
import os
import pandas as pd

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


@app.route('/api/calculate-risk', methods=['POST'])
def calculate_risk():
    """
    Calculate risk score for given conditions.
    Uses Kendall's compute_risk_score function.
    
    Expected JSON:
    {
        "hour": 18,
        "day_of_week": 5,
        "is_weekend": 0,
        "month": 12,
        "time_of_day": 3,
        "road_type": "residential",
        "road_surface": "wet",
        "PRCP": 0.5,
        "SNOW": 0,
        "TAVG": 32,
        "weather_category": "rain",
        "alcohol_related": 0,
        "is_freezing": 1
    }
    
    Returns:
        JSON with risk score and status
    """
    try:
        from model_loader import compute_risk_score
        
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No data provided'}), 400
        
        # Convert to DataFrame
        input_df = pd.DataFrame([data])
        
        # Calculate risk score
        risk_score = compute_risk_score(input_df)
        
        if risk_score is None:
            return jsonify({
                'error': 'Models not loaded. Please run crash_prediction.py first to train and save models.'
            }), 503
        
        return jsonify({
            'risk_score': risk_score,
            'status': 'success'
        })
        
    except Exception as e:
        return jsonify({
            'error': str(e),
            'status': 'error'
        }), 500


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

