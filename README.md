# Predicting and Visualizing Traffic Crash Risk in Tacoma, WA

Project for CPTS 437 to predict and visualize traffic crash likelihood in Tacoma, WA, using machine learning and an Agile workflow.

## 1. Problem Statement & Motivation

Traffic crashes in Tacoma, Washington, are a persistent safety concern, resulting in injuries, delays, and significant economic losses. While historical crash data is collected, it is often underutilized for predictive, forward-looking insights.

This project aims to bridge that gap by developing a system that:
1.  **Predicts** crash likelihood on Tacoma roads using historical crash, weather, and road data.
2.  **Visualizes** this risk through an interactive, map-based web interface for the public.

This is a practical application of machine learning to a real-world safety problem. It provides a valuable tool for transportation officials and the public while giving our team hands-on experience in data analysis, modeling, and full-stack web development.

## 2. Expected Features (Expected Results)

* **Crash Risk Predictor:** A trained machine learning model (or set of models) that inputs conditions (time, weather, location type) and outputs a risk score.
* **Model Evaluation Report:** A comparison of our models using metrics like Accuracy, Precision, Recall, F1-Score, and ROC/AUC curves.
* **Interactive Risk Map:** A web-based heatmap of Tacoma, allowing users to see which areas and conditions have a higher predicted risk.
* **Feature Importance Analysis:** A short report identifying which features (e.g., rain, time of day, road type) are most predictive of a crash.

## 3. Tech Stack & Key Packages

* **Data Analysis & Modeling:** Python
    * `pandas`: For data ingestion, cleaning, and manipulation.
    * `scikit-learn`: For feature engineering, model training (Logistic Regression, Decision Trees), and evaluation.
* **Backend & Integration:** Python
    * `Flask` / `FastAPI`: To build a lightweight API that serves model predictions and the web frontend.
* **Data Visualization:** Python & JS
    * `Folium` (Python): For generating interactive Leaflet.js maps (heatmaps, markers).
    * `Leaflet.js` (JavaScript): The underlying library for the interactive map.
* **Frontend:** `HTML5`, `CSS3`
* **Core C++ Component:** A small C++ module for a specific algorithm component (e.g., a decision tree split), demonstrating language interoperability.

## 4. Project Workflow (Data Flow)

This diagram outlines the flow of data from its raw state to the end-user visualization.



1.  **Data Ingestion:** Raw `.csv` collision records from WSDOT and historical weather data from NOAA are loaded.
2.  **Data Preprocessing (Python):** Scripts clean, merge, and transform the data. This includes handling missing values, joining datasets on date/location, and feature engineering.
3.  **Model Training (Python):** The cleaned, feature-engineered dataset is used to train and validate several predictive models.
4.  **Prediction Generation:** The best-performing model is saved and used to generate risk predictions.
5.  **Visualization (Python/Folium):** Risk scores are aggregated (e.g., by road segment or zip code) and used to generate a Folium heatmap layer.
6.  **Backend (Flask/FastAPI):** An API endpoint is created to serve the pre-generated Folium map (`index.html`) and potentially handle dynamic prediction requests in the future.
7.  **Frontend (HTML/CSS):** The user accesses a clean, styled web page that embeds and displays the interactive map.

## 5. Agile Methodology & Sprints

We are following a lightweight Agile methodology with four one-week sprints.

* **Sprint 1 – Data Preparation:**
    * **Goal:** Collect and clean all source data.
    * **Deliverable:** A single, cleaned, and merged dataset ready for modeling.
* **Sprint 2 – Modeling & C++ Component:**
    * **Goal:** Build and evaluate baseline predictive models.
    * **Deliverable:** A trained model file and an evaluation notebook. A separate C++ component implementing a core logic piece.
* **Sprint 3 – Visualization Prototype:**
    * **Goal:** Create a map-based visualization of the data.
    * **Deliverable:** A static `index.html` file containing a Folium heatmap of crash risk.
* **Sprint 4 – Integration & Frontend:**
    * **Goal:** Connect the model, data, and map in a polished web application.
    * **Deliverable:** A working web app served by Flask/FastAPI that displays the interactive risk map.

## 6. Modeling Approach

This section outlines our plan for feature engineering, model selection, and validation, incorporating feedback to enhance technical depth.

### 6.1. Feature Engineering

We will extract and encode features from the WSDOT and NOAA datasets.

* **Potential WSDOT Features:**
    * `Time of Day` (e.g., 'Morning Rush', 'Mid-day')
    * `Day of Week` (e.g., 'Weekday', 'Weekend')
    * `Road Type` (e.g., 'Highway', 'Arterial', 'Residential')
    * `Light Conditions` (e.g., 'Daylight', 'Dark - Lit', 'Dark - Unlit')
    * `Junction Type` (e.g., 'Intersection', 'Driveway', 'Not at Junction')
* **Potential NOAA Features:**
    * `Precipitation` (Continuous, inches)
    * `Temperature` (Continuous, °F)
    * `Visibility` (Continuous, miles)
    * `Weather Condition` (Categorical, e.g., 'Clear', 'Rain', 'Snow', 'Fog')

### 6.2. Data Encoding

To prepare data for modeling, we will use:
* **One-Hot Encoding:** For nominal categorical features (e.g., `Road Type`, `Weather Condition`) to convert them into a machine-readable format without implying an order.
* **Label Encoding / Binning:** For ordinal features (e.g., `Light Conditions`) or to bin continuous features like `Time of Day`.

### 6.3. Model Selection & Validation

Given this is a classification problem (Crash / No-Crash), we will start with simple, interpretable models and use robust validation techniques.

* **Models:**
    1.  **Logistic Regression:** As a strong, interpretable baseline.
    2.  **Decision Tree:** To visualize the "rules" that lead to a crash.
    3.  **Random Forest:** (Stretch Goal) An ensemble model to improve predictive power.

* **Validation Strategy:**
    * **K-Fold Cross-Validation:** We will use 10-fold cross-validation to get a reliable estimate of model performance and ensure it generalizes to unseen data.
    * **Evaluation Metrics:** Because crash data is highly **imbalanced** (many "No-Crash" instances, few "Crash" instances), accuracy alone is misleading. We will focus on:
        * **Precision & Recall:** To understand the trade-off between false positives and false negatives.
        * **F1-Score:** The harmonic mean of Precision and Recall.
        * **ROC/AUC Curve:** To visualize the model's performance at all classification thresholds.

## 7. Limitations & Future Work

### 7.1. Project Limitations

* **Data Bias:** Crash data is subject to **under-reporting**, especially for non-injury or minor incidents. Our model's predictions are only as good as the data provided.
* **Spatial Autocorrelation:** Crashes are not random, independent events; they cluster. Our current models (like Logistic Regression) assume independence, which may be a source of error.
* **Real-Time Prediction:** This system is a **retrospective analysis**. A true real-time prediction system would require a live data pipeline, which is outside the current scope.

### 7.2. Future Work & Scalability

* **Deployment:** The Flask/FastAPI backend can be **containerized using Docker** and deployed to a cloud platform (like AWS or Heroku) to create a persistent, public-facing dashboard.
* **Advanced Models:** Future iterations could explore geospatial models (to account for spatial autocorrelation) or time-series models (like LSTMs) to better capture temporal patterns.
* **API Integration:** The frontend could be enhanced by integrating with a public API like **Google Maps** or **Mapbox** to allow users to get risk-based routing.


## 8. Team Roles & Contributors

* **Julia:** Data Ingestion & Cleaning (Python)
* **Kendall:** Modeling & Evaluation (Python / C++)
* **Zach:** Map Visualization (Python + HTML/CSS)
* **Zuriel:** Web Frontend & Integration (HTML/CSS + Python)
