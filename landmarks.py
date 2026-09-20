"""
Real coordinates for the landmarks that appear in combined_hotel_data.csv's
nearest_landmark column. Sourced from Wikipedia/public geographic records.

A few landmarks (marked below) don't have a precisely documented public
location (e.g. a specific "city centre" reference point some listings use
instead of a named place) - these use the city's well-established central
coordinate as the closest honest approximation, noted inline.
"""

LANDMARK_COORDS = {
    # Karnataka (Bangalore)
    "Bangalore Palace": (12.9987, 77.5920),
    "Ulsoor Lake": (12.9815, 77.6205),
    "MG Road": (12.9757, 77.6068),
    "Cubbon Park": (12.9763, 77.5929),
    "Indira Gandhi Musical Fountain Park": (12.9634, 77.5855),
    "Kempegowda Airport ": (13.1986, 77.7066),
    "Bangalore city centre": (12.9716, 77.5946),  # city-centre reference point
    "Palace Grounds": (13.0027, 77.5883),

    # Tamil Nadu (Chennai)
    "city centre": (13.0827, 80.2707),  # Chennai city-centre reference point
    "Thousand Lights Mosque": (13.0602, 80.2528),
    "Chennai Airport": (12.9941, 80.1709),
    "Chennai Egmore Railway Station": (13.0778, 80.2610),
    "Chennai Central Railway Station": (13.0827, 80.2757),
    "Marina Beach": (13.0500, 80.2824),

    # Delhi
    "Delhi city centre": (28.6139, 77.2090),  # Delhi city-centre reference point
    "T1 - Delhi Airport (IGI Airport)": (28.5562, 77.0850),
    "Indira Gandhi International Airport": (28.5562, 77.1000),
    "Metro Museum": (28.6280, 77.2190),  # Delhi Metro Museum, Patel Chowk
    "India Gate": (28.6129, 77.2295),
    "Connaught Place": (28.6315, 77.2167),
    "Red Fort": (28.6562, 77.2410),
    "New Delhi Railway Station": (28.6435, 77.2197),
    "BLK Hospital": (28.6436, 77.1796),
    "Agrasen Ki Baoli": (28.6262, 77.2246),

    # Telangana (Hyderabad)
    "Rajiv Gandhi International Airport": (17.2403, 78.4294),
    "Secunderabad Junction Railway Station ": (17.4344, 78.5017),

    # West Bengal (Kolkata)
    "Park Street": (22.5535, 88.3524),
    "Eden Gardens": (22.5646, 88.3433),
    "Netaji Subhash Chandra Bose International Airport": (22.6547, 88.4467),
    "Acropolis Mall": (22.5140, 88.3931),
    "Salt Lake Stadium": (22.5646, 88.4088),
    "Salt Lake Sec V": (22.5761, 88.4315),
    "Dharamtala Market": (22.5626, 88.3528),
    "Kalighat Kali Temple": (22.5192, 88.3428),
    "College Street": (22.5734, 88.3639),

    # Maharashtra (Mumbai)
    "T1 - Chhatrapati Shivaji International Airport": (19.0896, 72.8656),
    "Cooper hospital": (19.0996, 72.8367),
    "T2 - Chhatrapati Shivaji International Airport": (19.0896, 72.8656),
    "Queen's Necklace - Marine Drive": (18.9432, 72.8234),
    "Band Stand": (19.0433, 72.8202),
    "Tata memorial cancer hospital": (19.0049, 72.8432),
    "Gateway Of India Mumbai": (18.9220, 72.8347),
    "Siddhivinayak Temple": (19.0170, 72.8302),
    "Mumbai Central Bus Terminus": (18.9711, 72.8194),
}

# State -> list of its landmark names, in the order they should appear in
# the app's dropdown (kept in the same order the data naturally groups them).
STATE_LANDMARKS = {
    "Karnataka": [
        "Bangalore Palace", "Ulsoor Lake", "MG Road", "Cubbon Park",
        "Indira Gandhi Musical Fountain Park", "Kempegowda Airport ",
        "Bangalore city centre", "Palace Grounds",
    ],
    "Tamil Nadu": [
        "city centre", "Thousand Lights Mosque", "Chennai Airport",
        "Chennai Egmore Railway Station", "Chennai Central Railway Station",
        "Marina Beach",
    ],
    "Delhi": [
        "Delhi city centre", "T1 - Delhi Airport (IGI Airport)",
        "Indira Gandhi International Airport", "Metro Museum", "India Gate",
        "Connaught Place", "Red Fort", "New Delhi Railway Station",
        "BLK Hospital", "Agrasen Ki Baoli",
    ],
    "Telangana": [
        "Rajiv Gandhi International Airport",
        "Secunderabad Junction Railway Station ",
    ],
    "West Bengal": [
        "Park Street", "Eden Gardens",
        "Netaji Subhash Chandra Bose International Airport", "Acropolis Mall",
        "Salt Lake Stadium", "Salt Lake Sec V", "Dharamtala Market",
        "Kalighat Kali Temple", "College Street",
    ],
    "Maharashtra": [
        "T1 - Chhatrapati Shivaji International Airport", "Cooper hospital",
        "T2 - Chhatrapati Shivaji International Airport",
        "Queen's Necklace - Marine Drive", "Band Stand",
        "Tata memorial cancer hospital", "Gateway Of India Mumbai",
        "Siddhivinayak Temple", "Mumbai Central Bus Terminus",
    ],
}
