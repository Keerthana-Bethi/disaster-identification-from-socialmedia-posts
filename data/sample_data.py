"""
Data handling for disaster identification system.
"""
import pandas as pd
import os
import random

# URLs for stock images from pre-fetched list
DISASTER_SCENES = [
    "https://pixabay.com/get/g8bfe7d7f7a752e2775462923bf9b40c0538ba3b81a9569b1fdc8714fcbe3f64358b0937eb1cbdee66e52afe0f323bd9117087fa79ebd7204d9c0ff8ad6001967_1280.jpg",
    "https://pixabay.com/get/g39c5a201c8899452ac8fdf3255c09d23b30d210ab7d16c47985b8cb13b85c4501d40187299f604e2c5ace72bf4b280626e26027ffefdd82c75d3f74ace0fac8f_1280.jpg",
    "https://pixabay.com/get/g4b575304296836dcfb1977127c52f62ebb8288a6e683d6ce662814ec4af462d256dce9b9cffe723c0749f449344f1268708a4c6f24eef6bb2467acfa3b6ab1c5_1280.jpg",
    "https://pixabay.com/get/g59c679ef576ce3be1ca9ae3ad3affcb10a0e0be8f6602411e8bf04e0ce9777429b95526fa49de0e51f113ab0cc0f09a418b34424bec3bc093dd7880ac83570ca_1280.jpg",
    "https://pixabay.com/get/g94ec614aed7012c9bf34029bfe5d46cfcf9684a19f86bf51e793ac92e8a4aa3f4e1fc37494d704f9803ed4369a4f9747f46b8e6488f5b90078932038528e3696_1280.jpg",
    "https://pixabay.com/get/g8a6d0d60df753e1dba19446c2ee5ca9d650de2ab0a557dde9d9eec9afab6f9749431c1590e0d46b5bc7854ca49076f4b8f7cd601cdabe9aebb9891ce99b4cd8e_1280.jpg"
]

FLOOD_IMAGES = [
    "https://pixabay.com/get/g7c57ef3751294414257f49aa165b453732bdaf58e098641668f12b29a21b5527d13262c415e73f75616f6abd1f7ce1585423dbe3fe6eb6cd8267a1f6094e728f_1280.jpg",
    "https://pixabay.com/get/g55c37554e99a80d418f6d4cd818d427491cb8acea983f52900b15a099a28d7c6431e0fc322b20a22b7526ac2d2a95ca46712e6ab591ea14a2e01cc476c65e2d4_1280.jpg",
    "https://pixabay.com/get/g140c8de00cc1f2d97272936f81a9bfae62affacb9791cd2eea6854b0ff9f61de8041dae2a87590cdbf2f42b7355d61741122b0f60270f4e532171caba618d07b_1280.jpg",
    "https://pixabay.com/get/g4b8e342ee7886e7a5b781bc4d83cdbb934129485057ca296c3184025be006a3ca8ea516c93e8ff0d40fcc464849f797d062c123c11a6f4fd411f1eafaf963f15_1280.jpg"
]

FIRE_IMAGES = [
    "https://pixabay.com/get/gaf9af2ca0e6b2cbc3824955cbe79f1215fa78b55ed97be18ba5a012e61b300d6d5213d83ab9a52a6313038ebfcd89ea5f19279a2831591727f52922d7146ef26_1280.jpg",
    "https://pixabay.com/get/gb98313d25f325ce94671fed40bca6717cd0fd0fcd5f46e110c53e64b4fe90144d15197421aecc58fa6b97aad3cf0791a73434ea4d14d9671c1715951a27ef7d5_1280.jpg",
    "https://pixabay.com/get/g3a1f1b9f1ae4bceb82701eab4a47f0d102899a24ad4211a7a2fe91e2f7ddcb3c94eba2c21ef946464209cb0ef1505d38025441c1680c02e597023646613668d7_1280.jpg",
    "https://pixabay.com/get/g85198ed743bc1908da9def0d738aa82d6f64d7c843edd31d4700df8294e75b4b1364ea3ba60ea2d2e9612eafb0a53f4a8bfb7b7cb4e211e0af232c032d1aa943_1280.jpg"
]

# Sample tweet texts for each disaster type
SAMPLE_TWEETS = {
    "flood": [
        "Massive flooding in our area. Roads completely submerged. #flood #disaster",
        "Our neighborhood is underwater after last night's storm. Need help! #flooding #emergency",
        "The river has overflowed its banks, houses are flooded. Stay safe everyone. #flood #disaster",
        "Water levels rising rapidly. Evacuation orders in place. #flooding #emergency",
        "Flash floods have damaged several homes in the downtown area. #flood #disaster"
    ],
    "fire": [
        "Wildfire spreading quickly due to high winds. Several homes evacuated. #fire #disaster",
        "Massive fire at downtown building. Firefighters on scene. #fire #emergency",
        "Forest fire has destroyed thousands of acres. Air quality warnings issued. #wildfire #disaster",
        "House fire reported in residential area. Fire crews responding. #fire #emergency",
        "The flames are spreading fast! Everyone evacuate immediately! #fire #disaster"
    ],
    "earthquake": [
        "Strong tremors felt across the city. Buildings damaged. #earthquake #disaster",
        "Powerful earthquake just hit. Objects falling off shelves. #earthquake #emergency",
        "Aftershocks continue after major earthquake. Stay clear of damaged buildings. #earthquake #disaster",
        "7.2 magnitude earthquake reported. Tsunami warning issued. #earthquake #emergency",
        "Building collapsed after earthquake. Rescue teams searching for survivors. #earthquake #disaster"
    ],
    "hurricane": [
        "Hurricane approaching the coast. Mandatory evacuation ordered. #hurricane #disaster",
        "Powerful winds from the hurricane have damaged power lines. #hurricane #emergency",
        "Hurricane expected to make landfall tonight. Prepare immediately. #hurricane #disaster",
        "Storm surge from hurricane has flooded coastal areas. #hurricane #emergency",
        "Category 4 hurricane heading our way. Boarding up windows. #hurricane #disaster"
    ],
    "tornado": [
        "Tornado warning in effect. Take shelter immediately. #tornado #disaster",
        "Funnel cloud spotted near downtown. Seeking shelter now. #tornado #emergency",
        "Tornado has caused significant damage to homes and businesses. #tornado #disaster",
        "Multiple tornadoes reported in the area. Stay in your safe room. #tornado #emergency",
        "Tornado just passed through our neighborhood. Houses destroyed. #tornado #disaster"
    ],
    "not_disaster": [
        "Beautiful sunny day at the beach today! #weather #sunshine",
        "Just finished reading a great book. Highly recommended! #reading #books",
        "Traffic is heavy on the interstate this morning. Plan accordingly. #commute #traffic",
        "New restaurant opened downtown. The food is amazing! #foodie #restaurant",
        "Enjoying a peaceful walk in the park. So relaxing! #nature #outdoors"
    ]
}

# Combined dataset
def get_sample_dataset(n_samples=20):
    """
    Create a sample dataset for training/testing the model.
    
    Args:
        n_samples: Number of samples to generate
        
    Returns:
        Pandas DataFrame with images, text, and labels
    """
    data = []
    
    # Categories with their associated image lists
    categories = {
        "flood": FLOOD_IMAGES,
        "fire": FIRE_IMAGES,
        "earthquake": DISASTER_SCENES[:2],
        "hurricane": DISASTER_SCENES[2:4],
        "tornado": DISASTER_SCENES[4:],
        "not_disaster": []  # No specific images for non-disasters
    }
    
    # Create sample data
    for i in range(n_samples):
        # Randomly select a category
        category = random.choice(list(categories.keys()))
        
        # Select tweet text
        tweet = random.choice(SAMPLE_TWEETS[category])
        
        # Select image URL based on category
        if category == "not_disaster":
            # For non-disasters, just use a random URL
            img_url = random.choice(DISASTER_SCENES + FLOOD_IMAGES + FIRE_IMAGES)
            is_disaster = 0
        else:
            # For disasters, use an appropriate image
            img_url = random.choice(categories[category]) if categories[category] else random.choice(DISASTER_SCENES)
            is_disaster = 1
        
        # Create entry
        entry = {
            "image_url": img_url,
            "tweet_text": tweet,
            "category": category,
            "is_disaster": is_disaster  # Binary label (1 for disaster, 0 for not disaster)
        }
        
        data.append(entry)
    
    return pd.DataFrame(data)

def get_sample_demo_data():
    """Return sample data for demonstration purposes"""
    demo_data = [
        {
            "image_url": FLOOD_IMAGES[0],
            "tweet_text": "Massive flooding in our area. Roads completely submerged. #flood #disaster",
            "category": "flood",
            "is_disaster": 1
        },
        {
            "image_url": FIRE_IMAGES[0],
            "tweet_text": "Wildfire spreading quickly due to high winds. Several homes evacuated. #fire #disaster",
            "category": "fire", 
            "is_disaster": 1
        },
        {
            "image_url": DISASTER_SCENES[0],
            "tweet_text": "7.2 magnitude earthquake reported. Tsunami warning issued. #earthquake #emergency",
            "category": "earthquake",
            "is_disaster": 1
        },
        {
            "image_url": DISASTER_SCENES[3],
            "tweet_text": "Beautiful sunny day at the beach today! #weather #sunshine",
            "category": "not_disaster",
            "is_disaster": 0
        }
    ]
    return pd.DataFrame(demo_data)
