import requests

API_KEY = ""  # Replace with your actual Google Geolocation API key

def get_user_location():
    url = f"https://www.googleapis.com/geolocation/v1/geolocate?key={API_KEY}"
    payload = {"considerIp": "true"}  # Use IP-based location as fallback
    try:
        response = requests.post(url, json=payload, timeout=10)
        response.raise_for_status()  # Raise an error for bad status codes
        location = response.json()["location"]
        return location["lat"], location["lng"]
    except requests.exceptions.RequestException as e:
        raise Exception(f"Geolocation API error: {str(e)}")
    except KeyError:
        raise Exception("Invalid response format from Geolocation API")