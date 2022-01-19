import numpy as np

# group purposes
coarse_purpose_mapping = {
    "arts": "leisure",
    "church": "leisure",
    "nightlife": "leisure",
    "travel": "leisure",
    "vacation": "leisure",
    "outdoor_city": "leisure",
    "residential": "leisure",
    "restaurant": "leisure",
    "shop": "shop",
    "doctor": "shop",
    "home": "home",
    "office": "work",
    "school": "work",
    "sport": "leisure",
    "leisure": "leisure",
    "errand": "shop",
    "other": "unknown",
}

final_categories = ["work", "home", "leisure", "shop"]


def get_coarse_purpose_category(p):
    fine_cat = _get_fine_purpose_category(p)
    return coarse_purpose_mapping[fine_cat]


def one_hot_purpose(p):
    purpose_vector = np.zeros(len(final_categories))
    purpose_coarse = get_coarse_purpose_category(p)
    # all zeros if unknown purpose
    if purpose_coarse == "unknown":
        return purpose_vector
    # one hot vector
    purpose_vector[final_categories.index(purpose_coarse)] = 1
    return purpose_vector


def purpose_df_to_matrix(purpose_df):
    if "purpose" not in purpose_df.columns:
        return np.zeros((len(purpose_df), len(final_categories)))
    one_hot_array = np.array(
        [one_hot_purpose(p) for p in purpose_df["purpose"]]
    )
    assert len(one_hot_array) == len(purpose_df)
    return one_hot_array


# Fine grained purpose categories as they are appearing in tist
def _get_fine_purpose_category(p):
    low = p.lower()
    if (
        low == "office"
        or "conference" in low
        or "coworking" in low
        or "work" in low
    ):
        return "office"
    elif (
        "food" in low
        or "restaurant" in low
        or "pizz" in low
        or "salad" in low
        or "ice cream" in low
        or "bakery" in low
        or "burger" in low
        or "sandwich" in low
        or "caf" in low
        or "diner" in low
        or "snack" in low
        or "steak" in low
        or "pub" in low
        or "tea" in low
        or "noodle" in low
        or "chicken" in low
        or "brewery" in low
        or "breakfast" in low
        or "beer" in low
        or "bbq" in low
    ):
        return "restaurant"
    elif (
        "doctor" in low
        or "hospital" in low
        or "medical" in low
        or "emergency" in low
        or "dental" in low
        or "dentist" in low
    ):
        return "doctor"
    elif (
        "bus" in low
        or "airport" in low
        or "train" in low
        or "taxi" in low
        or "station" in low
        or "metro" in low
        or "travel" in low
        or "ferry" in low
    ):
        return "travel"
    elif (
        "store" in low
        or "shop" in low
        or "bank" in low
        or "deli" in low
        or "mall" in low
        or "arcade" in low
        or "boutique" in low
        or "post" in low
        or "market" in low
        or "dealership" in low
        or "errand" in low
    ):
        return "shop"
    elif (
        "bar" in low
        or "disco" in low
        or "club" in low
        or "nightlife" in low
        or "speakeasy" in low
    ):
        return "nightlife"
    elif "home" in low:
        return "home"
    elif "residential" in low or "building" in low or "neighborhood" in low:
        return "residential"
    elif (
        "entertain" in low
        or "theater" in low
        or "music" in low
        or "concert" in low
        or "museum" in low
        or "art" in low
        or "temple" in low
        or "historic" in low
    ):
        return "arts"
    elif (
        "golf" in low
        or "tennis" in low
        or "dance" in low
        or "sport" in low
        or "gym" in low
        or "hiking" in low
        or "skating" in low
        or "soccer" in low
        or "basketball" in low
        or "surf" in low
        or "stadium" in low
        or "baseball" in low
        or "yoga" in low
    ):
        return "sport"
    elif (
        "school" in low
        or "college" in low
        or "university" in low
        or "student" in low
    ):
        return "school"
    elif "church" in low or "mosque" in low or "spiritual" in low:
        return "church"
    elif (
        "vacation" in low
        or "hotel" in low
        or "beach" in low
        or "tourist" in low
        or "bed &" in low
    ):
        return "vacation"
    elif (
        "city" in low
        or "park" in low
        or "plaza" in low
        or "bridge" in low
        or "outdoors" in low
        or "playground" in low
        or "lake" in low
        or "pier" in low
        or "field" in low
        or "harbor" in low
    ):
        return "outdoor_city"
    elif low == "leisure":
        return "leisure"
    else:
        return "other"
