import pandas as pd
import logging
from utils.geolocation import get_user_location
from utils.distance import haversine

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    # Load dataset
    try:
        df = pd.read_csv("data/hospitals.csv")
        logger.info("Dataset loaded successfully")
    except FileNotFoundError:
        logger.error("hospitals.csv not found in data/ directory")
        print("❌ Error: hospitals.csv not found. Please ensure the file exists in the data/ directory.")
        return
    except Exception as e:
        logger.error(f"Error loading dataset: {e}")
        print(f"❌ Error: {e}")
        return

    # Validate required columns
    required_columns = ['Doctor_ID', 'Doctor_Name', 'Specialization', 'Experience_Years', 'Contact_No',
                       'Hospital_ID', 'Hospital_Name', 'Area', 'City', 'Rating', 'Lat', 'Lng']
    if not all(col in df.columns for col in required_columns):
        logger.error("Dataset missing required columns")
        print("❌ Error: Dataset missing required columns. Required: " + ", ".join(required_columns))
        return

    # Get user location
    try:
        user_lat, user_lng = get_user_location()
        logger.info(f"User location fetched: Lat={user_lat}, Lng={user_lng}")
        print(f"\n📍 User Location: {user_lat}, {user_lng}")
    except Exception as e:
        logger.error(f"Failed to get user location: {e}")
        print(f"❌ Failed to get live location: {e}. Please enter manually.")
        try:
            user_lat = float(input("Enter your latitude: "))
            user_lng = float(input("Enter your longitude: "))
        except ValueError:
            logger.error("Invalid manual location input")
            print("❌ Invalid input. Exiting.")
            return

    # Ask for disease/specialization (default to Pediatrics if not specified)
    specialization = input("\nEnter the disease or specialization (e.g., Pediatrics, press Enter for Pediatrics): ").strip()
    specialization = specialization if specialization else "Pediatrics"  # Default to Pediatrics
    logger.info(f"Requested specialization: {specialization}")

    # Filter dataset by specialization (case-insensitive)
    filtered_df = df[df['Specialization'].str.lower() == specialization.lower()]

    if filtered_df.empty:
        logger.warning(f"No specialists found for {specialization}")
        print(f"❌ No doctors found with specialization: {specialization}")
        return

    # Compute distances
    filtered_df['Distance_km'] = filtered_df.apply(
        lambda row: haversine(user_lat, user_lng, row['Lat'], row['Lng']),
        axis=1
    )
    logger.info("Distances calculated successfully")

    # Find nearest hospital
    nearest = filtered_df.sort_values(by='Distance_km').iloc[0]
    logger.info(f"Nearest hospital found: {nearest['Hospital_Name']} at {nearest['Distance_km']:.2f} km")

    # Display results
    print("\n✅ Nearest Hospital and Doctor Information:")
    print(f"🏥 Hospital: {nearest['Hospital_Name']} ({nearest['Area']}, {nearest['City']})")
    print(f"👨‍⚕️ Doctor: {nearest['Doctor_Name']} (ID: {nearest['Doctor_ID']})")
    print(f"📚 Specialization: {nearest['Specialization']}, Experience: {nearest['Experience_Years']} years")
    print(f"⭐ Rating: {nearest['Rating']}")
    print(f"📞 Contact: {nearest['Contact_No']}")
    print(f"📍 Distance: {nearest['Distance_km']:.2f} km")

    # Optionally display additional options if available
    if len(filtered_df) > 1:
        print("\n📋 Other Nearby Options:")
        for _, row in filtered_df.iloc[1:3].iterrows():  # Show top 2 alternatives
            print(f"🏥 {row['Hospital_Name']} ({row['Area']}, {row['City']}) - {row['Distance_km']:.2f} km")
            print(f"👨‍⚕️ {row['Doctor_Name']} (Exp: {row['Experience_Years']} yrs, Rating: {row['Rating']})")
            print(f"📞 {row['Contact_No']}\n")

if __name__ == "__main__":
    main()