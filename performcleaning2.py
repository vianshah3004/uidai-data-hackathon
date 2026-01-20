import pandas as pd
import numpy as np
import re
import os


def clean_data():
    try:
        script_dir = os.path.dirname(os.path.abspath(__file__))
    except NameError:
        script_dir = os.getcwd()

    print("Reading raw biometric data...")
    input_path = os.path.join(script_dir, "raw_biometric_data.csv")
    demographics = pd.read_csv(input_path)

    print("Cleaning dates...")
    demographics["date"] = pd.to_datetime(
        demographics["date"], format="%d-%m-%Y", errors="coerce"
    )
    demographics["date"] = demographics["date"].dt.strftime("%Y-%m-%d")

    df = demographics.copy()

    print("Dropping duplicates...")
    df = df.drop_duplicates()

    print("Cleaning States...")
    # State Cleaning
    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")

    df["state"] = df["state"].astype(str).str.strip().str.lower()

    df = df[df["state"] != "100000"]

    STATE_MAPPING = {
        "west bengal": "west bengal",
        "west  bengal": "west bengal",
        "west bangal": "west bengal",
        "west bengli": "west bengal",
        "westbengal": "west bengal",
        # Chhattisgarh
        "chhatisgarh": "chhattisgarh",
        "chhattisgarh": "chhattisgarh",
        # Odisha
        "orissa": "odisha",
        "odisha": "odisha",
        # Andaman & Nicobar
        "andaman & nicobar islands": "andaman and nicobar islands",
        "andaman and nicobar islands": "andaman and nicobar islands",
        # Dadra & Nagar Haveli and Daman & Diu
        "dadra & nagar haveli": "dadra and nagar haveli and daman and diu",
        "daman & diu": "dadra and nagar haveli and daman and diu",
        "dadra and nagar haveli": "dadra and nagar haveli and daman and diu",
        "daman and diu": "dadra and nagar haveli and daman and diu",
        "the dadra and nagar haveli and daman and diu": "dadra and nagar haveli and daman and diu",
        # Jammu & Kashmir
        "jammu & kashmir": "jammu and kashmir",
        "jammu and kashmir": "jammu and kashmir",
        # Puducherry
        "pondicherry": "puducherry",
        "puducherry": "puducherry",
        # Misplaced district/city names mapped to states
        "darbhanga": "bihar",
        "puttenahalli": "karnataka",
        "balanagar": "telangana",
        "uttaranchal": "uttarakhand",
        "jaipur": "rajasthan",
        "madanapalle": "andhra pradesh",
        "nagpur": "maharashtra",
        "raja annamalai puram": "tamil nadu",
        "r.a. puram": "tamil nadu",
        "tamilnadu": "tamil nadu",
    }

    df["state"] = df["state"].replace(STATE_MAPPING)
    df["state"] = df["state"].str.title()

    print("Dropping duplicates after state cleaning...")
    df = df.drop_duplicates()

    print("Cleaning Districts...")
    # District Cleaning
    df["district"] = df["district"].astype(str).str.strip().str.title()

    df["district"] = df["district"].str.replace(r"[*()]", "", regex=True).str.strip()

    # District Mappings
 

    mapping = {
        "Coochbehar": "Cooch Behar",
        "Hooghiy": "Hooghly",
        "Banas Kantha": "Banaskantha",
        "Dinajpur Uttar": "Uttar Dinajpur",
        "East Midnapur": "Purba Medinipur",
        "Debagarh": "Deogarh",
        "Jajapur": "Jajpur",
        "Khorda": "Khordha",
        "Sonapur": "Kamrup Metro",
        "Sundergarh": "Sundargarh",
        "Baleshwar": "Balasore",
        "Baleswar": "Balasore",
        "Pondicherry": "Puducherry",
        "Firozpur": "Ferozepur",
        "Shaheed Bhagat Singh Nagar": "Shahid Bhagat Singh Nagar",
        "Sribhumi": "Karimganj",
        "Gariyaband": "Gariaband",
        "Kawardha": "Kabirdham",
        "Kabeerdham": "Kabirdham",
        "Purbi Singhbhum": "East Singhbhum",
        "Purbi Singhbum": "East Singhbhum",
        "Bellary": "Ballari",
        "Gulbarga": "Kalaburagi",
        "Ramanagar": "Bengaluru South",
        "Kasargod": "Kasaragod",
        "West Nimar": "Khargone",
        "Ahmednagar": "Ahilyanagar",
        "Ahmadnagar": "Ahilyanagar",
        "Ahmed Nagar": "Ahilyanagar",
        "Anantapur": "Anantapuramu",
        "Ananthapur": "Anantapuramu",
        "Ananthapuramu": "Anantapuramu",
        "Cuddapah": "YSR",
        "Y. S. R": "YSR",
        "Anugul": "Angul",
        "Medchal-Malkajgiri": "Medchal Malkajgiri",
        "Medchal?Malkajgiri": "Medchal Malkajgiri",
        "Medchal−Malkajgiri": "Medchal Malkajgiri",
        "Yadadri.": "Yadadri Bhuvanagiri",
        "K.V.Rangareddy": "Ranga Reddy",
        "K.V. Rangareddy": "Ranga Reddy",
        "Rangareddi": "Ranga Reddy",
        "Karim Nagar": "Karimnagar",
        "Purbi Champaran": "East Champaran",
        "Marigaon": "Morigaon",
        "Dohad": "Dahod",
        "Pashchim Champaran": "West Champaran",
        "Kachchh": "Kutch",
        "Ashok Nagar": "Ashoknagar",
        "North Cachar Hills": "Dima Hasao",
        "Ahmadnagar": "Ahilyanagar",  
        "Jalor": "Jalore",
        "Kaimur Bhabua": "Kaimur",
        "Chhotaudepur": "Chhota Udepur",
        "Panchmahals": "Panchmahal",
        "Panch Mahals": "Panchmahal",
        "South Salmara Mankachar": "South Salmara-Mankachar",
        "Mahesana": "Mehsana",
        "Kancheepuram": "Kanchipuram",
        "Gurgaon": "Gurugram",
        "Sibsagar": "Sivasagar",
        "Bara Banki": "Barabanki",
        "K.V. Rangareddy": "Ranga Reddy",
        "Rangareddi": "Ranga Reddy",
        "Shrawasti": "Shravasti",
        "Arvalli": "Aravalli",
        "Surendra Nagar": "Surendranagar",
        "Punch": "Poonch",
        "South Sikkim": "Namchi",
        "West Sikkim": "Gyalshing",
        "The Dangs": "Dang",
        "Sas Nagar Mohali": "Mohali",
        "S.A.S Nagar": "Sahibzada Ajit Singh Nagar",
        "Chumukedima": "Dimapur",
        "Nabarangapu": "Nabarangpur",
        "Bangalore Rural": "Bengaluru Rural",
        "Baramula": "Baramulla",
        "Mewat": "Nuh",
        "Ahmadabad": "Ahmedabad",
        "24 Paraganas North": "North 24 Parganas",
        "Medinipur West": "Paschim Medinipur",
        "Gaurella Pendra Marwahi": "Gaurela-Pendra-Marwahi",
        "Visakhapatanam": "Visakhapatnam",
        "Medchal Malkajgiri": "Medchal-Malkajgiri",
        "Siddharth Nagar": "Siddharthnagar",
        "24 Paraganas South": "South 24 Parganas",
        "Sabar Kantha": "Sabarkantha",
        "East Singhbum": "East Singhbhum",
        "Thiruvallur": "Tiruvallur",
        "Dinajpur Dakshin": "Dakshin Dinajpur",
        "East Nimar": "Khandwa",
        "Kushi Nagar": "Kushinagar",
        "Mahbubnagar": "Mahabubnagar",
        "N. T. R": "NTR",
        "Sri Potti Sriramulu Nellore": "Nellore",
        "Kamrup Metro": "Kamrup Metropolitan",
        "Nabarangapur": "Nabarangpur",
        "Prayagraj": "Allahabad (Prayagraj)",
        "Ayodhya": "Faizabad (Ayodhya)",
        "Dharashiv": "Osmanabad (Dharashiv)",
        "Mahabub Nagar": "Mahabubnagar",
        "Nellore": "Spsr Nellore",
        "Warangal": "Warangal Urban",
        "Bagalkot": "Bagalkote",
        "Belgaum": "Belagavi",
        "Chamrajanagar": "Chamarajanagar",
        "Chamrajnagar": "Chamarajanagar",
        "Chickmagalur": "Chikkamagaluru",
        "Chikmagalur": "Chikkamagaluru",
        "Davangere": "Davanagere",
        "Bengaluru South": "Ramanagara",
        "Shimoga": "Shivamogga",
        "Tumkur": "Tumakuru",
        "Narmadapuram": "Narmada",
        "Narsimhapur": "Narsinghpur",
        "Buldana": "Buldhana",
        "Chhatrapati Sambhajinagar": "Chhatrapati Sambhajinagar",
        "Gondiya": "Gondia",
        "Nicobar": "Nicobars",
        "East Midnapore": "Purba Medinipur",
        "West Midnapore": "Paschim Medinipur",
        "Medinipur": "Paschim Medinipur",
        "West Medinipur": "Paschim Medinipur",
        "Barddhaman": "Purba Bardhaman",
        "Bardhaman": "Purba Bardhaman",
        "Burdwan": "Purba Bardhaman",
        "Hugli": "Hooghly",
        "Haora": "Howrah",
        "Hawrah": "Howrah",
        "Koch Bihar": "Cooch Behar",
        "Maldah": "Malda",
        "Puruliya": "Purulia",
        "South Dinajpur": "Dakshin Dinajpur",
        "North Dinajpur": "Uttar Dinajpur",
        "North Twenty Four Parganas": "North 24 Parganas",
        "South Twenty Four Parganas": "South 24 Parganas",
        "South 24 Pargana": "South 24 Parganas",
        "Bulandshahar": "Bulandshahr",
        "Sheikpura": "Sheikhpura",
        "Samstipur": "Samastipur",
        "Monghyr": "Munger",
        "Bhabua": "Kaimur",
        "Mumbai Sub Urban": "Mumbai Suburban",
        "Andamans": "South Andaman",
        "North Sikkim": "Mangan",
        "East Sikkim": "Gangtok",
        "West Sikkim": "Gyalshing",
        "South Sikkim": "Namchi",
        "East": "East Delhi",
        "North": "North Delhi",
        "West": "West Delhi",
        "South": "South Delhi",
        "North East": "North East Delhi",
        "RaigarhMh": "Raigad",
        "AurangabadBh": "Aurangabad",
        "Bandipur": "Bandipora",
        "Bandipore": "Bandipora",
        "Nawanshahr": "Shahid Bhagat Singh Nagar",
        "Sant Ravidas Nagar": "Bhadohi",
        "Sant Ravidas Nagar Bhadohi": "Bhadohi",
        "Jyotiba Phule Nagar": "Amroha",
        "Garhwal": "Pauri Garhwal",
        "Hardwar": "Haridwar",
        "Tuticorin": "Thoothukudi",
        "Leh Ladakh": "Leh",
        "Hanumakonda": "Warangal Urban",
        "Dakshin Bastar Dantewada": "Dantewada",
        "Pashchimi Singhbhum": "West Singhbhum",
        "Bid": "Beed",
        "Najafgarh": "South West Delhi",
        "S.A.S NagarMohali": "Mohali",
        "Tamulpur District": "Tamulpur",
        "Ahmednagar Ahilyanagar": "Ahilyanagar",
        "Sahibzada Ajit Singh Nagar Mohali": "Mohali",
        "Allahabad Prayagraj": "Allahabad (Prayagraj)",
        "Faizabad Ayodhya": "Faizabad (Ayodhya)",
        "Osmanabad Dharashiv": "Osmanabad (Dharashiv)",
        "Aurangabad Chhatrapati Sambhajinagar": "Chhatrapati Sambhajinagar",
        "Chatrapati Sambhaji Nagar": "Chhatrapati Sambhajinagar",
        # Spelling Corrections & Standards
        "Medchal Malkajgiri": "Medchal-Malkajgiri",
        "Manendragarh–Chirmiri–Bharatpur": "Manendragarh-Chirmiri-Bharatpur",
        "Yamuna Nagar": "Yamunanagar",
        "Badgam": "Budgam",
        "Hazaribag": "Hazaribagh",
        "Palamau": "Palamu",
        "Sahebganj": "Sahibganj",
        "Seraikela-Kharsawan": "Seraikela Kharsawan",
        "Chikkaballapur": "Chikballapur",
        "Baudh": "Boudh",
        "Jagatsinghapur": "Jagatsinghpur",
        "Chittaurgarh": "Chittorgarh",
        "Ganganagar": "Sri Ganganagar",
        "Jhunjhunun": "Jhunjhunu",
        "Kanniyakumari": "Kanyakumari",
        "The Nilgiris": "Nilgiris",
        "Thiruvarur": "Tiruvarur",
        "Thoothukkudi": "Thoothukudi",
        "Tirupattur": "Tirupathur",
        "Villupuram": "Viluppuram",
        "Jagitial": "Jagtial",
        "Jangoan": "Jangaon",
        "Komaram Bheem": "Kumuram Bheem",
        "Darjiling": "Darjeeling",
        "Purnea": "Purnia",
        "Gaurela-Pendra-Marwahi": "Gaurella-Pendra-Marwahi",
        "Khairagarh Chhuikhadan Gandai": "Khairagarh-Chhuikhadan-Gandai",
        "Mohla-Manpur-Ambagarh Chouki": "Mohla-Manpur-Ambagarh Chowki",
        "Mohalla-Manpur-Ambagarh Chowki": "Mohla-Manpur-Ambagarh Chowki",
        "Kodarma": "Koderma",
        "Pakaur": "Pakur",
        "Hasan": "Hassan",
        "Shupiyan": "Shopian",
        "Dhaulpur": "Dholpur",
        "Mammit": "Mamit",
        "Uttar Bastar Kanker": "Kanker",
        "Shi-Yomi": "Shi Yomi",
        "Janjgir - Champa": "Janjgir-Champa",
        "Janjgir Champa": "Janjgir-Champa",
        "Lahul And Spiti": "Lahaul and Spiti",
        "Lahul & Spiti": "Lahaul and Spiti",
        "Raebareli": "Rae Bareli",
        "Leparada": "Lepa Rada",
        "Tseminyu": "Tseminyü",
        "Mahrajganj": "Maharajganj",
        "Pakke Kessang": "Pakke-Kessang",
        "Jaintia Hills": "West Jaintia Hills",
        "Dadra & Nagar Haveli": "Dadra and Nagar Haveli",
        "Dadra And Nagar Haveli": "Dadra and Nagar Haveli",
        "Mumbai": "Mumbai City",
        "Bardez": "North Goa",
        "Sahibzada Ajit Singh Nagar": "Mohali",
        "Meluri": "Meluri",
        "Dr. B. R. Ambedkar Konaseema": "Dr. B. R. Ambedkar Konaseema",
        "Sri Muktsar Sahib": "Sri Muktsar Sahib",
        "Ranga Reddy": "Rangareddy",
        "Nellore": "Spsr Nellore",
        "Bangalore": "Bengaluru Urban",
        "Bengaluru": "Bengaluru Urban",
        "Mysore": "Mysuru",
        "Bengaluru South": "Ramanagara",
        "Ahilyanagar": "Ahilyanagar",
        "Osmanabad": "Osmanabad (Dharashiv)",
        "Shahid Bhagat Singh Nagar Nawanshahr": "SNawanshahr",
        "Shahid Bhagat Singh Nagar": "Nawanshahr",
        "Medchal Malkajgiri": "Medchal-Malkajgiri",
        "Allahabad": "Allahabad (Prayagraj)",
        "Faizabad": "Faizabad (Ayodhya)",
        "Kamrup Metro": "Kamrup Metropolitan",
        "Meluri": "Meluri",
        "Sri Muktsar Sahib": "Sri Muktsar Sahib",
        "Dr. B. R. Ambedkar Konaseema": "Dr. B. R. Ambedkar Konaseema",
        "Purba Champaran": "East Champaran",
        "Anugal": "Angul",
        "Rajauri": "Rajouri",
        "BijapurKar": "Vijayapura",
        "Bagpat": "Baghpat",
        "Sahibzada Ajit Singh Nagar (Mohali)": "Mohali",
        "Sahibzada Ajit Singh Nagar": "Mohali",
        "Mohali": "Mohali",
        "Medchal Malkajgiri": "Medchal-Malkajgiri",
        "Aurangabad (Chhatrapati Sambhajinagar)": "Chhatrapati Sambhajinagar",
        "Lahaul And Spiti": "Lahaul and Spiti",
        "South  Twenty Four Parganas": "South 24 Parganas",
        "naihati anandabazar": "North 24 Parganas",
        "South DumdumM": "North 24 Parganas",
        "Dist : Thane": "Thane",
        "Near University Thana": "Thane",
        "Nellore": "Spsr Nellore",
        "Manendragarhchirmiribharatpur": "Manendragarh-Chirmiri-Bharatpur",
        # Telangana
        "Idpl Colony": "Medchal Malkajgiri",
        "Medchalâ\x88\x92Malkajgiri": "Medchal Malkajgiri",
        "Medchal-Malkajgiri": "Medchal Malkajgiri",
        # Andhra Pradesh
        "Kadiri Road": "Sri Sathya Sai",
        "Rangareddy": "Ranga Reddy",
        # Odisha
        "BhadrakR": "Bhadrak",
        "Tiswadi": "North Goa",
        "Bicholim": "North Goa",
        "Bally Jagachha": "Howrah",
        "Domjur": "Howrah",
        "Balianta": "Khordha",
        "Naihati Anandabazar": "North 24 Parganas",
        "Near Uday Nagar Nit Garden": "Nagpur",
        "Near Dhyana Ashram": "Chennai",
        "5Th Cross": "Bengaluru Urban",
        "Near Meera Hospital": "Jaipur",
    }
    df = df[df["district"].astype(str) != "?"]

    # Apply mapping
    df["district"] = df["district"].replace(mapping)
    print(df["district"].unique())

    # Check for empty strings
    df = df[df["district"] != ""]
    print(df.duplicated().sum())
    df.drop_duplicates(inplace=True)
    print(df.duplicated().sum())
    print("Saving final cleaned data...")
    IMAGE_DIR = os.path.join(script_dir, "image")
    os.makedirs(IMAGE_DIR, exist_ok=True)
    output_file = os.path.join(IMAGE_DIR, "final_UIDAI_cleaned_biometrics_data.csv")
    df.to_csv(output_file, index=False)
    print(f"Successfully saved to {output_file}")

    # Print stats
    print(f"Final shape: {df.shape}")
    print(f"Unique Districts: {df['district'].nunique()}")


if __name__ == "__main__":
    clean_data()
