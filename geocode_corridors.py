"""
Geocode Tacoma Corridors and prepare for visualization
- Loads corridors from Corridors_(Tacoma).csv
- Geocodes each street to get lat/lon coordinates
- Saves geocoded corridors to corridors_geocoded.csv

Author: Zachary Marseglia
"""

import pandas as pd
from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut, GeocoderServiceError
import time

print("Loading corridors data...")
corridors = pd.read_csv('Corridors_(Tacoma).csv')
print(f"Found {len(corridors)} corridors")

# Initialize geocoder with a user agent
geolocator = Nominatim(user_agent="tacoma_crash_risk_viz")

def geocode_street(street_name, max_retries=3):
    """Geocode a street name in Tacoma, WA"""
    # Add Tacoma, WA to the street name for better results
    full_address = f"{street_name}, Tacoma, WA, USA"
    
    for attempt in range(max_retries):
        try:
            print(f"  Geocoding: {street_name}...", end=' ')
            location = geolocator.geocode(full_address, timeout=10)
            
            if location:
                print(f"✓ ({location.latitude:.4f}, {location.longitude:.4f})")
                return location.latitude, location.longitude
            else:
                print("x Not found")
                return None, None
                
        except (GeocoderTimedOut, GeocoderServiceError) as e:
            print(f"Retry {attempt + 1}/{max_retries}")
            time.sleep(2)
    
    print("x Failed after retries")
    return None, None

print("\nGeocoding corridors (this will take ~40 seconds)...")
latitudes = []
longitudes = []

for idx, row in corridors.iterrows():
    lat, lon = geocode_street(row['st_name'])
    latitudes.append(lat)
    longitudes.append(lon)
    
    if idx < len(corridors) - 1:  # Don't sleep after the last request
        time.sleep(1)

# Add coordinates to the dataframe
corridors['latitude'] = latitudes
corridors['longitude'] = longitudes

# Count successful geocodes
successful = corridors[corridors['latitude'].notna()].shape[0]
print(f"\n✓ Successfully geocoded {successful}/{len(corridors)} corridors")

# Save geocoded corridors
output_file = 'corridors_geocoded.csv'
corridors.to_csv(output_file, index=False)
print(f"✓ Saved geocoded corridors to: {output_file}")

# Show sample
print("\nSample geocoded corridors:")
print(corridors[['st_name', 'type', 'latitude', 'longitude']].head(10).to_string(index=False))
