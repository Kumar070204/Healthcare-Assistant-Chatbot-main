
import pandas as pd
import numpy as np
import re
from rapidfuzz import fuzz, process
import nltk
from sklearn.preprocessing import minmax_scale
nltk.download('punkt')
nltk.download('punkt_tab') # Download the missing resource


# --- Configuration / thresholds (tweak if needed) ---
CSV_PATH = r"C:\Users\Kumaraswamy\Downloads\Data_Quest (2)\Data_Quest\scripts\data\Indian_Food_Nutrition_Categorized_Broad_with_labels.csv"  # change if needed
KEYWORD_FUZZY_THRESHOLD = 80  # rapidfuzz score threshold for fuzzy match (0-100)
MAX_RETURN = 5

# --- Load dataset ---
df = pd.read_csv(CSV_PATH)
df.fillna('', inplace=True)

# Normalize and prepare fields
if 'machine_keywords' not in df.columns:
    # If old file, try to build from Diseases_Symptoms (fallback)
    df['machine_keywords'] = df['Diseases_Symptoms'].astype(str).str.lower().str.replace(r'[^a-z0-9|_ ]','',regex=True)

df['machine_keywords'] = df['machine_keywords'].astype(str).str.lower()
df['dish_name'] = df.get('Dish Name', df.columns[0]).astype(str)
# make a compact string description for explanations
descr_cols = [c for c in ['Broad_Category','Detailed_Category','Detailed Category','Detailed_Category.1'] if c in df.columns]
df['short_desc'] = df[descr_cols].astype(str).agg(' | '.join, axis=1)

# Collect label/confidence columns automatically (columns that end with _conf or known labels)
conf_cols = [c for c in df.columns if c.endswith('_conf')]
label_cols = [c[:-5] for c in conf_cols]  # e.g. Diabetes_conf -> Diabetes
# If none found, try a fallback list produced in preprocessing
if not conf_cols:
    # fallback trying to detect binary columns for common labels
    possible_labels = ['Diabetes','Hypertension','High_Cholesterol_HeartDisease','Obesity_HighCalorie','Low_Calorie_Option',
                       'High_Fiber_Benefit','Anemia_Iron_Folate_Benefit','Pregnancy_Friendly','Malnutrition_Protein_Benefit',
                       'GERD_Pancreatitis_Risk']
    conf_cols = [l + '_conf' for l in possible_labels if (l + '_conf') in df.columns]
    label_cols = [c[:-5] for c in conf_cols]

# Normalize numeric columns for use in scoring if present
num_cols = {}
for c in ['Calories (kcal)','Calories','Sodium (mg)','Sodium','Free Sugar (g)','Protein (g)','Fats (g)','Fibre (g)']:
    if c in df.columns:
        try:
            df[c] = pd.to_numeric(df[c], errors='coerce').fillna(0)
            num_cols[c] = df[c]
        except:
            pass

# scale some numeric features into 0-1 for scoring
scaled = {}
for key, series in num_cols.items():
    if series.max() > 0:
        scaled[key] = minmax_scale(series.values.astype(float))
    else:
        scaled[key] = series.values.astype(float)

# --- Utility functions: NLP parsing of user sentence ---
PREFERENCE_KEYWORDS = {
    'low_sodium': ['low salt','low sodium','less salt','reduced salt','no salt'],
    'low_calorie': ['low calorie','light meal','light food','low cal','low-calorie','lose weight','weight loss'],
    'high_protein': ['high protein','protein rich','more protein','protein'],
    'vegetarian': ['veg','vegetarian','no meat','plant-based','vegetarianism','only veg'],
    'vegan': ['vegan','no dairy','plant only'],
    'diabetic': ['diabetic','diabetes','high blood sugar','blood sugar','sugar level'],
    'hypertension': ['hypertension','high blood pressure','bp','high bp'],
    'pregnant': ['pregnant','pregnancy','expecting'],
    'no_spicy': ['not spicy','no spice','no spicy','less spicy','avoid spicy','not spicy please'],
    'gluten_free': ['gluten free','no gluten']
}
# reverse mapping for quick lookup
pref_map = {}
for k,v in PREFERENCE_KEYWORDS.items():
    for term in v:
        pref_map[term] = k

# Tamil personalization mapping (simple examples)
TAMIL_PERSONALIZATION = {
    'recommended': 'parinthuraikkappatta', # பரிந்துரைக்கப்பட்ட
    'for you': 'ungalukkaaga', # உங்களுக்காக
    'low calorie': 'kuraivaana kalori', # குறைவான கலோரி
    'high protein': 'athika puratham', # அதிக புரதம்
    'good for diabetes': 'sarkarai noyikku nallathu', # சர்க்கரை நோய்க்கு நல்லது
    'high sodium': 'athika sōṭiyam', # அதிக சோடியம்
    'avoid if': 'thavirka vēṇṭum eṉṟāl', # தவிர்க்க வேண்டும் என்றால்
    'calories': 'kalori', # கலோரி
    'protein': 'puratham', # புரதம்
    'sodium': 'sōṭiyam', # சோடியம்
    'sugar': 'sarkarai' # சர்க்கரை
}


def simple_preference_extract(text):
    text_l = text.lower()
    prefs = {}
    # direct phrase match
    for phrase, code in pref_map.items():
        if phrase in text_l:
            prefs[code] = prefs.get(code, 0) + 1
    # token-level hints
    words = nltk.word_tokenize(text_l)
    # a couple more heuristics
    if any(w in words for w in ['sugar','diabetic','diabetes']):
        prefs['diabetic'] = prefs.get('diabetic',0)+1
    if any(w in words for w in ['pregnant','pregnancy']):
        prefs['pregnant'] = prefs.get('pregnant',0)+1
    # detect "avoid X" patterns: "avoid spicy" etc -> map to no_spicy
    if re.search(r'avoid\s+\w+', text_l):
        m = re.search(r'avoid\s+(\w+)', text_l)
        if m:
            w = m.group(1)
            if w in ['spicy','salt','fat','sugar']:
                if w=='spicy':
                    prefs['no_spicy'] = prefs.get('no_spicy',0)+1
                if w=='salt':
                    prefs['low_sodium'] = prefs.get('low_sodium',0)+1
                if w=='sugar':
                    prefs['diabetic'] = prefs.get('diabetic',0)+1
    return prefs

# --- Keyword extraction & fuzzy matching ---
# Build a set of unique machine keywords across dataset
all_keywords = set()
for s in df['machine_keywords'].astype(str):
    for token in [t.strip() for t in s.split('|') if t.strip()]:
        all_keywords.add(token)
all_keywords = sorted(all_keywords)

# Map dish row indices to their machine keyword tokens
row_keywords = df['machine_keywords'].astype(str).str.split('|').apply(lambda tokens: [t.strip() for t in tokens if t.strip()])

def extract_keywords_from_sentence(sentence):
    s = sentence.lower()
    # Exact token matches
    exact_matches = set()
    for token in all_keywords:
        if token and ((' ' + token + ' ') in (' ' + s + ' ') or re.search(r'\b' + re.escape(token) + r'\b', s)):
            exact_matches.add(token)
    # Fuzzy matches (for small typos / forms)
    fuzzy_matches = set()
    # only run fuzzy for tokens not matched exactly and with reasonable length
    candidates = [t for t in all_keywords if len(t) >= 3 and t not in exact_matches]
    # Use process.extract from rapidfuzz for efficiency
    # Compare against the whole sentence and also compare word-by-word
    for token in candidates:
        score = fuzz.partial_ratio(token, s)
        if score >= KEYWORD_FUZZY_THRESHOLD:
            fuzzy_matches.add(token)
    matches = sorted(exact_matches.union(fuzzy_matches))
    return matches

# --- Scoring / ranking function ---
class RecommenderState:
    def __init__(self):
        self.last_sorted_idx = None
        self.pointer = 0

state = RecommenderState()

def score_dishes(matches, preferences):
    """
    Compute a score for each dish (row in df). Higher is better.
    Components:
      - keyword_score: sum of matched keyword confidences (from dish conf columns if available)
      - pref_boost: boost for preferences (e.g., low_calorie -> uses Low_Calorie_Option_conf)
      - penalty: penalize dishes contradictory to preferences (e.g., diabetic pref penalizes high sugar)
      - nutrition heuristics: mild penalties for high sodium if user prefers low_sodium
    Returns a pandas Series of scores (index matches df.index)
    """
    n = len(df)
    base_score = np.zeros(n, dtype=float)
    # Keyword based score
    if matches:
        for i, kws in enumerate(row_keywords):
            # count matches and factor by per-dish confidences where possible
            match_score = 0.0
            for m in matches:
                if m in kws:
                    # If the dish has a conf column for the matching label, use it. Otherwise add 0.6
                    # Many machine_keywords are generic tokens; we try to map token -> label_conf if available.
                    matched_conf = 0.6
                    # try to map m to a label_conf (e.g., 'diabetes' -> 'Diabetes_conf')
                    label_search = None
                    for lab in label_cols:
                        if lab.lower() in m or m in lab.lower() or lab.lower() in m.replace('_',' '):
                            label_search = lab + '_conf'
                            break
                    if label_search and label_search in df.columns:
                        try:
                            matched_conf = float(df.iloc[i][label_search])
                        except:
                            matched_conf = 0.6
                    match_score += matched_conf
            base_score[i] = match_score
    else:
        # If no explicit keywords, use general affinity (prefer low calorie if user asked weight-loss etc.)
        base_score = np.zeros(n, dtype=float)

    # Preference boosts/penalties
    pref_boost = np.zeros(n, dtype=float)
    penalty = np.zeros(n, dtype=float)

    # helper to safely get conf column or 0
    def conf(col, idx):
        return float(df.iloc[idx].get(col, 0)) if col in df.columns else 0.0

    for idx in range(n):
        # low_sodium preference: boost Low_Calorie_Option? No. We will penalize high sodium and boost Hypertension_conf low values if available.
        if 'low_sodium' in preferences:
            sod = df.iloc[idx].get('Sodium (mg)', df.iloc[idx].get('Sodium', 0))
            if not np.isnan(sod) and sod > 0:
                penalty[idx] += (sod / 2000.0) * 0.6  # scaled penalty
            # also boost dishes that explicitly are marked with Hypertension_conf low (i.e., low risk)
            if 'Hypertension_conf' in df.columns:
                pref_boost[idx] += max(0, 1 - conf('Hypertension_conf', idx)) * 0.3

        if 'low_calorie' in preferences:
            # boost Low_Calorie_Option_conf and penalize high calories
            if 'Low_Calorie_Option_conf' in df.columns:
                pref_boost[idx] += conf('Low_Calorie_Option_conf', idx) * 0.6
            cal = df.iloc[idx].get('Calories (kcal)', df.iloc[idx].get('Calories', 0))
            if not np.isnan(cal):
                penalty[idx] += (max(0, cal - 400) / 1000.0) * 0.5

        if 'high_protein' in preferences:
            if 'Malnutrition_Protein_Benefit_conf' in df.columns:
                pref_boost[idx] += conf('Malnutrition_Protein_Benefit_conf', idx) * 0.8
            # add small boost from absolute protein if available
            prot = df.iloc[idx].get('Protein (g)', df.iloc[idx].get('Protein', 0))
            if prot and not np.isnan(prot):
                pref_boost[idx] += min(1.0, prot / 20.0) * 0.3

        if 'diabetic' in preferences:
            # penalize high sugar strongly; boost low sugar/confidence
            sugar = df.iloc[idx].get('Free Sugar (g)', df.iloc[idx].get('Free Sugar (g)', 0))
            if sugar and not np.isnan(sugar):
                penalty[idx] += min(1.0, sugar / 20.0) * 0.9
            if 'Diabetes_conf' in df.columns:
                # prefer dishes with low diabetes_conf (meaning not flagged as bad) OR explicitly safe (we didn't create a "safe" column, so prefer low_conf)
                pref_boost[idx] += max(0, 1 - conf('Diabetes_conf', idx)) * 0.5

        if 'hypertension' in preferences:
            # penalize high sodium and prefer low Hypertension_conf
            sod = df.iloc[idx].get('Sodium (mg)', df.iloc[idx].get('Sodium', 0))
            if sod and not np.isnan(sod):
                penalty[idx] += min(1.0, sod / 1500.0) * 0.8
            if 'Hypertension_conf' in df.columns:
                pref_boost[idx] += max(0, 1 - conf('Hypertension_conf', idx)) * 0.5

        if 'pregnant' in preferences:
            if 'Pregnancy_Friendly_conf' in df.columns:
                pref_boost[idx] += conf('Pregnancy_Friendly_conf', idx) * 0.9
            # prefer good iron/folate presence
            if 'Anemia_Iron_Folate_Benefit_conf' in df.columns:
                pref_boost[idx] += conf('Anemia_Iron_Folate_Benefit_conf', idx) * 0.5

        if 'vegetarian' in preferences or 'vegan' in preferences:
            # boost vegetarian/vegan by checking Broad_Category / Detailed_Category
            bc = str(df.iloc[idx].get('Broad_Category','')).lower()
            if 'vegetarian' in preferences and ('veg' in bc or 'vegetarian' in bc):
                pref_boost[idx] += 0.6
            if 'vegan' in preferences and ('veg' in bc or 'vegetarian' in bc or 'plant' in bc):
                pref_boost[idx] += 0.6
            # small penalty for meat categories
            if any(x in bc for x in ['meat','chicken','mutton','fish','egg']):
                penalty[idx] += 0.6

        if 'no_spicy' in preferences:
            # we don't have spice info; apply small penalty to entries that have 'spicy' keyword
            if 'spicy' in df.iloc[idx]['machine_keywords']:
                penalty[idx] += 0.6

    # Compose final score: base_score (keyword-driven) normalized + pref_boost - penalty
    # Normalize base_score
    if base_score.max() > 0:
        base_norm = base_score / (base_score.max() + 1e-9)
    else:
        base_norm = base_score
    final_score = base_norm * 0.6 + pref_boost * 0.5 - penalty * 0.6
    # also add a small tie-breaker by pref_boost and dish popularity proxies (if Calories present, prefer moderate calories)
    # clip final_score to sensible range
    final_score = np.nan_to_num(final_score)
    # scale to 0..1
    if final_score.max() > final_score.min():
        final_score = (final_score - final_score.min()) / (final_score.max() - final_score.min() + 1e-9)
    else:
        final_score = np.clip(final_score, 0, 1)
    return pd.Series(final_score, index=df.index)

# --- Recommendation API functions ---
def recommend(user_sentence, top_k=5):
    """
    Main function to call.
    Returns top_k recommendations in detailed sentence format and stores state for paging.
    """
    prefs = simple_preference_extract(user_sentence)
    keyword_matches = extract_keywords_from_sentence(user_sentence)

    # If user input has no keywords, fallback: use preferences to rank
    scores = score_dishes(keyword_matches, prefs)

    sorted_idx = scores.sort_values(ascending=False).index
    state.last_sorted_idx = sorted_idx
    state.pointer = top_k

    top_idx = sorted_idx[:top_k]
    results = []
    for rank, idx in enumerate(top_idx, start=1):
        row = df.loc[idx]
        dish = row.get('dish_name', row.get('Dish Name', 'Unknown'))
        reasons = []

        # Explanation based on user preferences and dish properties
        explanation_parts = []

        # Incorporate Tamil personalization
        explanation_parts.append(f"{dish}: {TAMIL_PERSONALIZATION.get('recommended', 'Recommended')} {TAMIL_PERSONALIZATION.get('for you', 'for you')}.")

        # Explain based on matched keywords
        matched_kws_for_dish = [k for k in row_keywords.iloc[idx] if k in keyword_matches]
        if matched_kws_for_dish:
            explanation_parts.append(f"It {TAMIL_PERSONALIZATION.get('matches your mention of', 'matches your mention of')}: {', '.join(matched_kws_for_dish)}.")

        # Explain based on preferences and relevant nutritional info/labels
        if 'diabetic' in prefs:
            sugar = row.get('Free Sugar (g)', row.get('Free Sugar (g)', 0))
            if not pd.isna(sugar):
                 explanation_parts.append(f"This dish has {sugar:.2f}g of {TAMIL_PERSONALIZATION.get('sugar', 'sugar')}. Foods low in {TAMIL_PERSONALIZATION.get('sugar', 'sugar')} are {TAMIL_PERSONALIZATION.get('good for diabetes', 'good for diabetes')}.")
            if 'Diabetes_conf' in df.columns and row['Diabetes_conf'] < 0.5:
                 explanation_parts.append(f"It is flagged with a low {TAMIL_PERSONALIZATION.get('diabetes', 'diabetes')} risk ({row['Diabetes_conf']:.2f}).")


        if 'low_calorie' in prefs:
            cal = row.get('Calories (kcal)', row.get('Calories', 0))
            if not pd.isna(cal):
                explanation_parts.append(f"With {cal:.2f} {TAMIL_PERSONALIZATION.get('calories', 'calories')}, this can be a {TAMIL_PERSONALIZATION.get('low calorie', 'low calorie')} option.")

        if 'high_protein' in prefs:
            prot = row.get('Protein (g)', row.get('Protein', 0))
            if not pd.isna(prot):
                explanation_parts.append(f"It provides {prot:.2f}g of {TAMIL_PERSONALIZATION.get('protein', 'protein')}, which is a good amount for {TAMIL_PERSONALIZATION.get('high protein', 'high protein')} needs.")

        if 'low_sodium' in prefs or 'hypertension' in prefs:
            sod = row.get('Sodium (mg)', row.get('Sodium', 0))
            if not pd.isna(sod):
                explanation_parts.append(f"This dish has {sod:.2f}mg of {TAMIL_PERSONALIZATION.get('sodium', 'sodium')}. {TAMIL_PERSONALIZATION.get('Avoid if', 'Avoid if')} you have {TAMIL_PERSONALIZATION.get('hypertension', 'hypertension')} and high {TAMIL_PERSONALIZATION.get('sodium', 'sodium')} is a concern.")
            if 'Hypertension_conf' in df.columns and row['Hypertension_conf'] < 0.5:
                 explanation_parts.append(f"It is flagged with a low {TAMIL_PERSONALIZATION.get('hypertension', 'hypertension')} risk ({row['Hypertension_conf']:.2f}).")


        # Add other relevant nutritional info if available and significant
        fibre = row.get('Fibre (g)', 0)
        if not pd.isna(fibre) and fibre > 3:
             explanation_parts.append(f"It's high in fiber ({fibre:.2f}g).")

        fats = row.get('Fats (g)', 0)
        if not pd.isna(fats) and fats > 15:
             explanation_parts.append(f"Note the fat content ({fats:.2f}g).")


        # Add top positive label confidences if not already covered
        top_label_conf_pairs = []
        for confc in conf_cols:
            val = float(row.get(confc,0))
            label_name = confc.replace('_conf','')
            # Avoid repeating if already explained by specific preference
            if label_name.lower() not in ['diabetes', 'low_calorie_option', 'malnutrition_protein_benefit', 'hypertension', 'pregnancy_friendly']:
                if val > 0.35:
                    top_label_conf_pairs.append((label_name, round(val,2)))

        if top_label_conf_pairs:
            explanation_parts.append("Other nutritional notes: " + ", ".join([f"{lab}({conf})" for lab,conf in top_label_conf_pairs[:3]]))


        full_explanation = " ".join(explanation_parts)

        results.append({
            'rank': rank,
            'dish': dish,
            'explanation': full_explanation,
            'score': float(scores.loc[idx]),
            'row_index': int(idx)
        })
    return results

def recommend_next(top_k=5):
    """
    Return the next top_k recommendations (fresh/new).
    """
    if state.last_sorted_idx is None:
        return []
    start = state.pointer
    end = start + top_k
    sorted_idx = state.last_sorted_idx
    next_idx = sorted_idx[start:end]
    state.pointer = end
    results = []
    for offset, idx in enumerate(next_idx, start=1):
        row = df.loc[idx]
        dish = row.get('dish_name', row.get('Dish Name', 'Unknown'))
        # Recompute score for next set if needed, but for simplicity here, just provide basic info
        # A more complex implementation would re-run score_dishes on the next_idx subset
        explanation = f"{dish}: Additional {TAMIL_PERSONALIZATION.get('recommended', 'Recommended')}."
        results.append({
            'rank': start + offset,
            'dish': dish,
            'explanation': explanation,
            'row_index': int(idx)
        })
    return results

# --- User Interaction ---
def get_recommendations():
    while True:
        user_input = input("What do you feel like eating? ")
        if not user_input:
            print("Please tell me what you'd like to eat to get recommendations.")
            continue

        recs = recommend(user_input, top_k=MAX_RETURN)

        if not recs:
            print("Sorry, I couldn't find any recommendations based on your request.")
        else:
            print("\nTop recommendations for you:\n")
            for r in recs:
                print(f"{r['rank']}. {r['dish']} -- {r['explanation']}\n")

        while True:
            follow_up = input("Do you want different recommendations (type 'more'), have another query (type 'query'), or exit (type 'exit')? ").lower().strip()
            if follow_up == 'more':
                next_recs = recommend_next(MAX_RETURN)
                if next_recs:
                    print("\nNext recommendations:\n")
                    for r in next_recs:
                        print(f"{r['rank']}. {r['dish']} -- {r['explanation']}\n")
                else:
                    print("No more recommendations available.")
            elif follow_up == 'query':
                break # break inner loop to ask for new query
            elif follow_up == 'exit':
                print("Exiting. Enjoy your meal!")
                return # exit outer loop and function
            else:
                print("Invalid input. Please type 'more', 'query', or 'exit'.")


# --- Main execution ---
if __name__ == "__main__":
    get_recommendations()