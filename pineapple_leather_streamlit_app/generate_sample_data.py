import numpy as np
import pandas as pd

SEED = 42
rng = np.random.default_rng(SEED)
N = 2500

personas = [
    "Eco-conscious premium adopters",
    "Design-first urban buyers",
    "Value-seeking curious buyers",
    "Skeptical traditional buyers",
    "Conscious gifting buyers",
]
persona_probs = np.array([0.22, 0.20, 0.24, 0.19, 0.15])
persona = rng.choice(personas, size=N, p=persona_probs)

age_map = {
    "Eco-conscious premium adopters": ["18-24", "25-34", "35-44"],
    "Design-first urban buyers": ["18-24", "25-34", "35-44"],
    "Value-seeking curious buyers": ["18-24", "25-34", "35-44", "45-54"],
    "Skeptical traditional buyers": ["25-34", "35-44", "45-54", "55+"],
    "Conscious gifting buyers": ["25-34", "35-44", "45-54"],
}

city_map = {
    "Eco-conscious premium adopters": ["Metro", "Tier 1", "Tier 2"],
    "Design-first urban buyers": ["Metro", "Tier 1"],
    "Value-seeking curious buyers": ["Tier 1", "Tier 2", "Tier 3", "Metro"],
    "Skeptical traditional buyers": ["Tier 2", "Tier 3", "Rural", "Tier 1"],
    "Conscious gifting buyers": ["Metro", "Tier 1", "Tier 2"],
}

regions = ["North", "South", "East", "West", "Central", "Northeast"]
region_probs = {
    "Eco-conscious premium adopters": [0.20, 0.22, 0.12, 0.20, 0.10, 0.16],
    "Design-first urban buyers": [0.18, 0.24, 0.10, 0.24, 0.10, 0.14],
    "Value-seeking curious buyers": [0.20, 0.18, 0.14, 0.18, 0.16, 0.14],
    "Skeptical traditional buyers": [0.22, 0.16, 0.16, 0.16, 0.18, 0.12],
    "Conscious gifting buyers": [0.20, 0.18, 0.12, 0.18, 0.12, 0.20],
}

occupations = ["Student", "Salaried", "Business Owner", "Homemaker", "Freelancer", "Creative Professional"]
occupation_probs = {
    "Eco-conscious premium adopters": [0.18, 0.40, 0.10, 0.05, 0.10, 0.17],
    "Design-first urban buyers": [0.16, 0.38, 0.10, 0.04, 0.10, 0.22],
    "Value-seeking curious buyers": [0.20, 0.42, 0.08, 0.09, 0.10, 0.11],
    "Skeptical traditional buyers": [0.08, 0.36, 0.16, 0.20, 0.08, 0.12],
    "Conscious gifting buyers": [0.10, 0.34, 0.12, 0.12, 0.10, 0.22],
}

income_levels = ["No income", "Below 25k", "25k-50k", "50k-100k", "100k-200k", "Above 200k"]
income_probs = {
    "Eco-conscious premium adopters": [0.06, 0.10, 0.18, 0.34, 0.22, 0.10],
    "Design-first urban buyers": [0.08, 0.10, 0.20, 0.34, 0.18, 0.10],
    "Value-seeking curious buyers": [0.12, 0.22, 0.30, 0.24, 0.09, 0.03],
    "Skeptical traditional buyers": [0.08, 0.24, 0.30, 0.24, 0.10, 0.04],
    "Conscious gifting buyers": [0.06, 0.14, 0.22, 0.30, 0.18, 0.10],
}

purchase_freq_levels = ["Very often", "Monthly", "Once in 2-3 months", "Rarely", "Almost never"]
purchase_freq_probs = {
    "Eco-conscious premium adopters": [0.20, 0.34, 0.26, 0.15, 0.05],
    "Design-first urban buyers": [0.22, 0.36, 0.24, 0.14, 0.04],
    "Value-seeking curious buyers": [0.08, 0.20, 0.34, 0.28, 0.10],
    "Skeptical traditional buyers": [0.05, 0.14, 0.28, 0.34, 0.19],
    "Conscious gifting buyers": [0.08, 0.18, 0.30, 0.30, 0.14],
}

channels = ["Brand website", "E-commerce marketplace", "Shopping mall/store", "Boutique/designer store", "Instagram/social commerce", "Exhibitions/pop-ups"]
channel_probs = {
    "Eco-conscious premium adopters": [0.18, 0.20, 0.18, 0.16, 0.12, 0.16],
    "Design-first urban buyers": [0.14, 0.22, 0.18, 0.18, 0.20, 0.08],
    "Value-seeking curious buyers": [0.10, 0.36, 0.26, 0.06, 0.14, 0.08],
    "Skeptical traditional buyers": [0.06, 0.20, 0.42, 0.06, 0.08, 0.18],
    "Conscious gifting buyers": [0.10, 0.24, 0.22, 0.12, 0.12, 0.20],
}

purchase_drivers = ["Quality/Durability", "Design/Aesthetics", "Brand Image", "Sustainability", "Price/Value", "Exclusivity"]
driver_probs = {
    "Eco-conscious premium adopters": [0.22, 0.16, 0.07, 0.34, 0.12, 0.09],
    "Design-first urban buyers": [0.18, 0.34, 0.16, 0.10, 0.10, 0.12],
    "Value-seeking curious buyers": [0.18, 0.14, 0.04, 0.14, 0.42, 0.08],
    "Skeptical traditional buyers": [0.34, 0.10, 0.08, 0.06, 0.36, 0.06],
    "Conscious gifting buyers": [0.18, 0.18, 0.10, 0.20, 0.12, 0.22],
}

mat_prefs = ["Genuine leather", "Vegan leather", "Fabric/Canvas", "Recycled material", "Natural handcrafted material", "No strong preference"]
mat_probs = {
    "Eco-conscious premium adopters": [0.10, 0.30, 0.08, 0.18, 0.26, 0.08],
    "Design-first urban buyers": [0.20, 0.26, 0.10, 0.10, 0.16, 0.18],
    "Value-seeking curious buyers": [0.18, 0.18, 0.18, 0.08, 0.10, 0.28],
    "Skeptical traditional buyers": [0.44, 0.10, 0.12, 0.05, 0.10, 0.19],
    "Conscious gifting buyers": [0.16, 0.20, 0.10, 0.10, 0.20, 0.24],
}

brand_styles = ["Luxury and premium", "Sustainable and conscious", "Minimal and modern", "Handcrafted and artisanal", "Youthful and trendy"]
brand_probs = {
    "Eco-conscious premium adopters": [0.18, 0.42, 0.14, 0.16, 0.10],
    "Design-first urban buyers": [0.30, 0.10, 0.28, 0.08, 0.24],
    "Value-seeking curious buyers": [0.10, 0.18, 0.22, 0.14, 0.36],
    "Skeptical traditional buyers": [0.16, 0.08, 0.20, 0.28, 0.28],
    "Conscious gifting buyers": [0.16, 0.20, 0.18, 0.28, 0.18],
}

story_options = ["Agri-waste innovation", "Supports Northeast farmers", "Sustainable luxury", "Cruelty-free innovation", "Modern premium Indian roots"]
story_probs = {
    "Eco-conscious premium adopters": [0.18, 0.24, 0.32, 0.16, 0.10],
    "Design-first urban buyers": [0.10, 0.12, 0.22, 0.18, 0.38],
    "Value-seeking curious buyers": [0.20, 0.18, 0.18, 0.10, 0.34],
    "Skeptical traditional buyers": [0.12, 0.12, 0.08, 0.08, 0.60],
    "Conscious gifting buyers": [0.16, 0.26, 0.18, 0.10, 0.30],
}

experience_options = ["Online store", "Marketplace", "Premium offline store", "Pop-up exhibition", "Designer boutique", "Social media page"]
experience_probs = {
    "Eco-conscious premium adopters": [0.18, 0.12, 0.20, 0.24, 0.16, 0.10],
    "Design-first urban buyers": [0.14, 0.18, 0.18, 0.10, 0.22, 0.18],
    "Value-seeking curious buyers": [0.18, 0.34, 0.18, 0.08, 0.08, 0.14],
    "Skeptical traditional buyers": [0.08, 0.18, 0.38, 0.12, 0.06, 0.18],
    "Conscious gifting buyers": [0.10, 0.16, 0.24, 0.22, 0.10, 0.18],
}

use_cases = ["Daily personal use", "Office/professional use", "Fashion/styling", "Gifting", "Special occasions", "Unlikely to buy"]
use_probs = {
    "Eco-conscious premium adopters": [0.26, 0.20, 0.18, 0.12, 0.12, 0.12],
    "Design-first urban buyers": [0.16, 0.18, 0.36, 0.10, 0.14, 0.06],
    "Value-seeking curious buyers": [0.22, 0.16, 0.18, 0.18, 0.10, 0.16],
    "Skeptical traditional buyers": [0.16, 0.14, 0.10, 0.10, 0.10, 0.40],
    "Conscious gifting buyers": [0.12, 0.10, 0.14, 0.40, 0.16, 0.08],
}

willing_levels = ["0%", "Up to 5%", "6%-10%", "11%-20%", "Above 20%"]
willing_probs = {
    "Eco-conscious premium adopters": [0.06, 0.16, 0.26, 0.34, 0.18],
    "Design-first urban buyers": [0.10, 0.18, 0.28, 0.28, 0.16],
    "Value-seeking curious buyers": [0.22, 0.34, 0.24, 0.14, 0.06],
    "Skeptical traditional buyers": [0.40, 0.28, 0.16, 0.10, 0.06],
    "Conscious gifting buyers": [0.14, 0.26, 0.28, 0.20, 0.12],
}

offer_options = ["First-time buyer discount", "Combo/bundle offer", "Limited edition packaging", "Free shipping", "Loyalty discount", "Customization offer"]
offer_probs = {
    "Eco-conscious premium adopters": [0.12, 0.20, 0.16, 0.08, 0.16, 0.28],
    "Design-first urban buyers": [0.10, 0.20, 0.22, 0.10, 0.12, 0.26],
    "Value-seeking curious buyers": [0.22, 0.30, 0.06, 0.20, 0.16, 0.06],
    "Skeptical traditional buyers": [0.26, 0.24, 0.06, 0.22, 0.16, 0.06],
    "Conscious gifting buyers": [0.12, 0.26, 0.20, 0.10, 0.12, 0.20],
}

product_cols = [
    "interest_handbags", "interest_wallets", "interest_belts", "interest_footwear",
    "interest_tech_accessories", "interest_apparel", "interest_home_decor", "interest_corporate_gifts"
]
trust_cols = [
    "trust_premium_design", "trust_durability", "trust_water_resistance", "trust_luxury_branding",
    "trust_certification", "trust_made_in_india", "trust_farmer_story", "trust_reasonable_pricing"
]
concern_cols = [
    "concern_durability", "concern_quality", "concern_low_awareness", "concern_high_price",
    "concern_brand_trust", "concern_appearance", "concern_availability"
]

def choice_from_map(label, mapping):
    return rng.choice(mapping[label], p=np.ones(len(mapping[label]))/len(mapping[label]))

def weighted_choice(label, options, prob_map):
    return rng.choice(options, p=prob_map[label])

rows = []
for i, p in enumerate(persona, start=1):
    age = choice_from_map(p, age_map)
    city = choice_from_map(p, city_map)
    region = weighted_choice(p, regions, region_probs)
    occupation = weighted_choice(p, occupations, occupation_probs)
    income = weighted_choice(p, income_levels, income_probs)
    freq = weighted_choice(p, purchase_freq_levels, purchase_freq_probs)
    channel = weighted_choice(p, channels, channel_probs)
    driver = weighted_choice(p, purchase_drivers, driver_probs)
    material = weighted_choice(p, mat_prefs, mat_probs)
    brand = weighted_choice(p, brand_styles, brand_probs)
    story = weighted_choice(p, story_options, story_probs)
    experience = weighted_choice(p, experience_options, experience_probs)
    use_case = weighted_choice(p, use_cases, use_probs)
    willing = weighted_choice(p, willing_levels, willing_probs)
    offer = weighted_choice(p, offer_options, offer_probs)

    gender = rng.choice(["Male", "Female", "Non-binary", "Prefer not to say"], p=[0.48, 0.46, 0.03, 0.03])

    sustain_base = {
        "Eco-conscious premium adopters": 4.4,
        "Design-first urban buyers": 3.5,
        "Value-seeking curious buyers": 3.2,
        "Skeptical traditional buyers": 2.1,
        "Conscious gifting buyers": 3.6,
    }[p]
    sustainability_importance = int(np.clip(round(rng.normal(sustain_base, 0.7)), 1, 5))

    awareness_base = {
        "Eco-conscious premium adopters": 3.7,
        "Design-first urban buyers": 3.2,
        "Value-seeking curious buyers": 2.6,
        "Skeptical traditional buyers": 1.8,
        "Conscious gifting buyers": 2.9,
    }[p]
    awareness_score = int(np.clip(round(rng.normal(awareness_base, 0.8)), 1, 4))
    awareness_labels = {
        1: "No, never heard of it",
        2: "Heard the term only",
        3: "Yes, a little",
        4: "Yes, know it well",
    }
    awareness = awareness_labels[awareness_score]

    familiarity = int(np.clip(round(rng.normal(awareness_score + 0.3, 0.8)), 1, 5))

    openness_base = {
        "Eco-conscious premium adopters": 4.3,
        "Design-first urban buyers": 4.0,
        "Value-seeking curious buyers": 3.4,
        "Skeptical traditional buyers": 2.1,
        "Conscious gifting buyers": 3.7,
    }[p] + (0.12 * (sustainability_importance - 3)) + (0.10 * (awareness_score - 2.5))
    openness = int(np.clip(round(rng.normal(openness_base, 0.8)), 1, 5))

    durability_base = {
        "Eco-conscious premium adopters": 3.4,
        "Design-first urban buyers": 3.1,
        "Value-seeking curious buyers": 2.9,
        "Skeptical traditional buyers": 2.4,
        "Conscious gifting buyers": 3.0,
    }[p]
    perceived_durability = int(np.clip(round(rng.normal(durability_base, 0.8)), 1, 5))

    recommend = int(np.clip(round(rng.normal(openness + 0.1, 0.9)), 1, 5))

    income_score = income_levels.index(income)
    freq_score = {"Almost never": 1, "Rarely": 2, "Once in 2-3 months": 3, "Monthly": 4, "Very often": 5}[freq]
    willing_score = {"0%": 0, "Up to 5%": 1, "6%-10%": 2, "11%-20%": 3, "Above 20%": 4}[willing]
    city_score = {"Rural": 1, "Tier 3": 2, "Tier 2": 3, "Tier 1": 4, "Metro": 5}[city]

    spend_score = (
        16 * income_score + 8 * freq_score + 7 * openness + 6 * willing_score + 4 * sustainability_importance + 3 * city_score + rng.normal(0, 10)
    )
    if p == "Skeptical traditional buyers":
        spend_score -= 8
    if p == "Design-first urban buyers":
        spend_score += 6
    spend_score = int(np.clip(round(spend_score), 20, 100))

    if spend_score < 35:
        small_budget = "Below 1000"
        large_budget = "Below 2000"
    elif spend_score < 50:
        small_budget = "1000-2500"
        large_budget = "2000-5000"
    elif spend_score < 65:
        small_budget = "2501-5000"
        large_budget = "5001-10000"
    elif spend_score < 80:
        small_budget = "5001-10000"
        large_budget = "10001-20000"
    else:
        small_budget = "Above 10000"
        large_budget = "Above 20000"

    product_probs = {
        "interest_handbags": 0.52,
        "interest_wallets": 0.48,
        "interest_belts": 0.32,
        "interest_footwear": 0.38,
        "interest_tech_accessories": 0.34,
        "interest_apparel": 0.22,
        "interest_home_decor": 0.20,
        "interest_corporate_gifts": 0.24,
    }
    persona_shift = {
        "Eco-conscious premium adopters": {"interest_handbags": 0.10, "interest_wallets": 0.04, "interest_home_decor": 0.10, "interest_corporate_gifts": 0.06},
        "Design-first urban buyers": {"interest_handbags": 0.14, "interest_footwear": 0.12, "interest_apparel": 0.10, "interest_tech_accessories": 0.06},
        "Value-seeking curious buyers": {"interest_wallets": 0.08, "interest_belts": 0.06, "interest_corporate_gifts": 0.04},
        "Skeptical traditional buyers": {"interest_handbags": -0.06, "interest_wallets": -0.04, "interest_home_decor": -0.04, "interest_corporate_gifts": -0.04},
        "Conscious gifting buyers": {"interest_corporate_gifts": 0.18, "interest_home_decor": 0.10, "interest_handbags": 0.02},
    }
    products = {}
    for col in product_cols:
        prob = product_probs[col] + persona_shift.get(p, {}).get(col, 0) + 0.02 * (openness - 3)
        products[col] = int(rng.random() < np.clip(prob, 0.03, 0.95))
    if sum(products.values()) == 0:
        products[rng.choice(product_cols)] = 1

    trust_base = {
        "trust_premium_design": 0.46 + 0.05 * (brand == "Luxury and premium") + 0.05 * (brand == "Minimal and modern"),
        "trust_durability": 0.44 + 0.06 * (driver == "Quality/Durability"),
        "trust_water_resistance": 0.30,
        "trust_luxury_branding": 0.25 + 0.08 * (brand == "Luxury and premium"),
        "trust_certification": 0.34 + 0.08 * (sustainability_importance >= 4),
        "trust_made_in_india": 0.28 + 0.06 * (story == "Modern premium Indian roots"),
        "trust_farmer_story": 0.24 + 0.08 * (story == "Supports Northeast farmers"),
        "trust_reasonable_pricing": 0.38 + 0.10 * (driver == "Price/Value"),
    }
    trusts = {col: int(rng.random() < np.clip(trust_base[col] + 0.03 * (openness - 3), 0.05, 0.9)) for col in trust_cols}
    if sum(trusts.values()) < 2:
        for c in rng.choice(trust_cols, size=2, replace=False):
            trusts[c] = 1

    concern_base = {
        "concern_durability": 0.34 + 0.08 * (perceived_durability <= 2),
        "concern_quality": 0.28 + 0.06 * (awareness_score <= 2),
        "concern_low_awareness": 0.26 + 0.12 * (awareness_score <= 2),
        "concern_high_price": 0.26 + 0.08 * (income_score <= 2),
        "concern_brand_trust": 0.22 + 0.10 * (awareness_score <= 2),
        "concern_appearance": 0.16 + 0.08 * (material == "Genuine leather"),
        "concern_availability": 0.18 + 0.05 * (city_score <= 2),
    }
    concerns = {col: int(rng.random() < np.clip(concern_base[col] + (0.03 if p == "Skeptical traditional buyers" else 0), 0.02, 0.9)) for col in concern_cols}

    trust_sum = sum(trusts.values())
    concern_sum = sum(concerns.values())
    intent_score = (
        0.75 * openness + 0.45 * sustainability_importance + 0.25 * awareness_score + 0.20 * trust_sum
        - 0.30 * concern_sum + 0.15 * willing_score + 0.12 * freq_score + rng.normal(0, 0.7)
    )
    if p == "Skeptical traditional buyers":
        intent_score -= 0.8
    if p == "Eco-conscious premium adopters":
        intent_score += 0.4

    if intent_score < 2.2:
        purchase_intent_5level = "Definitely would not buy"
    elif intent_score < 3.0:
        purchase_intent_5level = "Probably would not buy"
    elif intent_score < 3.8:
        purchase_intent_5level = "Not sure"
    elif intent_score < 4.6:
        purchase_intent_5level = "Probably would buy"
    else:
        purchase_intent_5level = "Definitely would buy"

    purchase_intent_binary = 1 if purchase_intent_5level in ["Probably would buy", "Definitely would buy"] else 0

    row = {
        "respondent_id": i,
        "hidden_persona": p,
        "age_group": age,
        "gender": gender,
        "city_tier": city,
        "region": region,
        "occupation": occupation,
        "income_range": income,
        "accessory_purchase_frequency": freq,
        "preferred_shopping_channel": channel,
        "premium_purchase_driver": driver,
        "sustainability_importance": sustainability_importance,
        "prior_eco_purchase": rng.choice(["Yes, frequently", "Yes, occasionally", "Heard of them but never purchased", "No"], p=[0.12, 0.34, 0.28, 0.26] if p != "Skeptical traditional buyers" else [0.04, 0.14, 0.24, 0.58]),
        "plant_based_leather_awareness": awareness,
        "familiarity_with_sustainable_materials": familiarity,
        "preferred_material": material,
        "openness_to_try": openness,
        "perceived_durability_vs_leather": perceived_durability,
        "preferred_first_experience": experience,
        "intended_use_case": use_case,
        "preferred_finish": rng.choice(["Classic leather-like", "Matte natural", "Glossy luxury", "Textured artisan", "Minimal modern"], p=[0.24, 0.20, 0.16, 0.16, 0.24]),
        "preferred_brand_style": brand,
        "preferred_brand_story": story,
        "usual_budget_small_accessory": small_budget,
        "usual_budget_large_accessory": large_budget,
        "willingness_to_pay_extra": willing,
        "preferred_offer": offer,
        "likelihood_to_recommend": recommend,
        "estimated_spending_score": spend_score,
        "purchase_intent_5level": purchase_intent_5level,
        "purchase_intent_binary": purchase_intent_binary,
    }
    row.update(products)
    row.update(trusts)
    row.update(concerns)
    rows.append(row)


df = pd.DataFrame(rows)

# Light controlled noise
noise_idx = rng.choice(df.index, size=int(0.03 * len(df)), replace=False)
df.loc[noise_idx[:20], "estimated_spending_score"] = np.clip(df.loc[noise_idx[:20], "estimated_spending_score"] + rng.integers(10, 18, 20), 20, 100)
df.loc[noise_idx[20:40], "purchase_intent_binary"] = 1 - df.loc[noise_idx[20:40], "purchase_intent_binary"]
# keep 5-level mostly aligned; leave as mild imperfection.

# Save
output = "sample_training_data.csv"
df.to_csv(output, index=False)
print(f"Saved {output} with shape {df.shape}")
print(df.head())
