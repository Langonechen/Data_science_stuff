
# Libraries:

from IPython.display import display
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay, classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# Getting the training data:
train_data = []
with open('train.jsonl', 'r') as file:
    for line in file:
        train_data.append(json.loads(line))

print(f"Successfully loaded {len(train_data)} battles")

# Get the test data:
test_data = []
with open('test.jsonl', 'r') as f:
    for line in f:
        test_data.append(json.loads(line))
print(f"Successfully loaded {len(test_data)} battles")

# Looking into the data:
first_battle = train_data[0]    #first entire row

# Copy the first battle and truncate the timeline for better display of data
battle_for_display = first_battle.copy()
battle_for_display['battle_timeline'] = battle_for_display.get('battle_timeline', [])[:6] #first 6 turns

# json.dumps for cleaner printing
print(json.dumps(battle_for_display, indent=4))


# Feature selection method
def feature_extraction(data:list[dict]) -> pd.DataFrame:
    
    """
    from the baseline (static features):
    a)  {In p1_team are there included certain pokemons that are very strong in gen 1}
    b)  {Critical hits!}

    from the timeline(dynamic features): 
    1. {p1_team and p2_team mean potential damage during the battle}
    2. {p1 and p2 team mean hp during the 30 rounds}
    3. {p1_team effective damage dealt from moves and/or caused by effects}
    4. {p1_team usage of induced change of status in the opponent's pokemon}
    5. {pokemon with a bad status for p1 and p2 in the last turn}
    """

    winning_features = []
    for battle in data:
        features = {}   # dictionary type
        
        # a) --- In p1_team are there certain pokemons that are considered strong ---
        best_p1_team = battle.get('p1_team_details', [])
        if best_p1_team:
            # Check for each target Pokemon in the team
            features['has_Alakazam'] = int(any(q.get('name', '').lower() == 'alakazam' for q in best_p1_team))
            features['has_Tauros'] = int(any(q.get('name', '').lower() == 'tauros' for q in best_p1_team))
            features['has_Snorlax'] = int(any(q.get('name', '').lower() == 'snorlax' for q in best_p1_team))
            features['has_Chansey'] = int(any(q.get('name', '').lower() == 'chansey' for q in best_p1_team))

        # b) --- Critical hits! ---
        # We calculate critical hits potential based on p1 pokemon speed stats
        p1_team = battle.get('p1_team_details', [])
        if p1_team:
            team_speeds = [s.get('base_spd', 0) for s in p1_team]
            # Gen 1 critical hits calculation: T = base_speed/2, critical hits occur if random(0,255) < T
            crit_rates = []
            for speed in team_speeds:
                T = speed/2
                crit_probability = min(T/256, 1.0)  # probability = T/256
                crit_rates.append(crit_probability)
            
            # critical hit features:
            features['p1_team_max_crit_rate'] = max(crit_rates) if crit_rates else 0  # because a critical hit is usually a rare event!
            features['p1_team_avg_crit_rate'] = sum(crit_rates)/len(crit_rates) if crit_rates else 0


        # --- DYNAMIC FEATURES --- 
        timeline = battle.get('battle_timeline', [])
        
        # initialize counters for damaging moves:
        base_power1 = 0
        base_power2 = 0
        
        p1_names = []
        p2_names = []
        
        # 1. --- p1_team and p2_team mean potential damange during battles ---
        # get their base power from moves
        for i in range(len(timeline)):
            x = timeline[i].get('p1_move_details')
            if (x):
                base_power1+=(x.get("base_power", 0))
            else:
                base_power1+=0

            x = timeline[i].get('p2_move_details')
            if (x):
                base_power2+=(x.get("base_power", 0))
            else:
                base_power2+=0

            # get their names
            x = timeline[i].get('p1_pokemon_state')
            if (x):
                p1name = x.get("name")
                
            x = timeline[i].get('p2_pokemon_state')
            if (x):
                p2name = x.get("name")
            p1_names.append(p1name)
            p2_names.append(p2name)
        
        p1_names = list(set(p1_names))
        p2_names = list(set(p2_names))
        
        # get their health points
        p1hp = {}
        p2hp = {}
        for el in p1_names:
            p1hp[el] = 100 #points
        for el in p2_names:
            p2hp[el] = 100 #points
        
        #
        for i in range(len(timeline)):
            x = timeline[i].get('p1_pokemon_state')
            if (x):
                p1hp[x.get("name")] = x.get("hp_pct")*100

            x = timeline[i].get('p2_pokemon_state')
            if (x):
                p2hp[x.get("name")] = x.get("hp_pct")*100
            
        base_power1/=len(timeline)
        features['p1_mean_potential_damage'] = base_power1
        base_power2/=len(timeline)
        features['p2_mean_potential_damage'] = base_power2

        # 2. --- p1 and p2 team mean hp during the 30 rounds --- 
        vl1 = 0
        for el in p1hp.keys():
            vl1 += p1hp[el] 

        vl2 = 0
        for el in p2hp.keys():
            vl2 += p2hp[el] 

        vl1 = vl1 + (6-len(p1hp.keys()))*100
        vl2 = vl2 + (6-len(p2hp.keys()))*100

        vl1/=6
        vl2/=6

        features['p1_r30_mean_hpt'] = vl1
        features['p2_r30_mean_hpt'] = vl2

        # initialize counters pt2:
        damage_dealt = 0
        change_of_status = 0
        
        # 3. --- p1_team mean effetive damage dealt ---
        for event in range(len(timeline)):
            p2_state = timeline[event].get('p2_pokemon_state')
            if p2_state:
                current_hp_pct = p2_state.get('hp_pct', float)
                if current_hp_pct < 1.0:
                    damage_dealt += (1.0 - current_hp_pct)

        # 4. --- p1_team usage of induced change of status in the opponent's pokemon ---
        for event in range(len(timeline)):
            p2_state = timeline[event].get('p2_pokemon_state')
            if p2_state:
                current_status = p2_state.get('status', '')
                if current_status != 'nostatus':
                    change_of_status += 1

        features['p1_effectively_registered_damage'] = damage_dealt
        features['p1_change_of_status_induced'] = change_of_status

        # 5. --- a pokemon with a bad status for p1 and p2 in the last turn ---
        if timeline:
            last_turn = timeline[-1]
            p1_status = last_turn['p1_pokemon_state']['status']
            p2_status = last_turn['p2_pokemon_state']['status']
            features['p1_bad_status'] = 1 if p1_status not in ['nostatus', 'noeffect'] else 0
            features['p2_bad_status'] = 1 if p2_status not in ['nostatus', 'noeffect'] else 0     

        # Battle id and target variable:
        features['battle_id'] = battle.get('battle_id')
        if 'player_won' in battle:
            features['player_won'] = int(battle['player_won'])
        
        winning_features.append(features)
    
    return pd.DataFrame(winning_features).fillna(0)


train_df = feature_extraction(train_data)
test_df = feature_extraction(test_data)
display(train_df.head())
display(test_df.head())

# Prepare features and target for validation split
features = [col for col in train_df.columns if col not in ['battle_id', 'player_won']]
X_features = train_df[features]
Y_target = train_df['player_won']

# Split the training data for validation
X_train, X_test, Y_train, Y_test = train_test_split(X_features, Y_target, test_size = 0.3, random_state = 42)

# Create a pipeline with preprocessing and classifier
clf = Pipeline([
    ('scaler', StandardScaler()),
    ('classifier', LogisticRegression(random_state = 42, max_iter = 1000))
])

# Train the model
clf.fit(X_features, Y_target)

# Make predictions on validation set
Y_predicted = clf.predict(X_test)
Y_probabilities = clf.predict_proba(X_test)[:, 1]

# Evaluate the model(Logistic)
accuracy = accuracy_score(Y_test, Y_predicted)

print(f"\nValidation Performance:")
print(f"Accuracy: {accuracy:.4f}")

# Plot confusion matrix
cm = confusion_matrix(Y_test, Y_predicted)
ConfusionMatrixDisplay(cm, display_labels=['Lost', 'Won']).plot(cmap='Blues')
plt.title("Confusion Matrix")
plt.show()


# Random Forest 
# Prepare features and target for validation split
features = [col for col in train_df.columns if col not in ['battle_id', 'player_won']]
X_features = train_df[features]
Y_target = train_df['player_won']

# Split the training data for validation
X_train, X_test, Y_train, Y_test = train_test_split(X_features, Y_target, test_size = 0.3, random_state = 42)

rf = RandomForestClassifier(
    n_estimators=200,  
    max_depth=10, 
    min_samples_leaf= 2,
    min_samples_split= 5,
    max_leaf_nodes=6,
    bootstrap=True
)

rf.fit(X_features, Y_target)
Y_pred = rf.predict(X_test)

print(classification_report(Y_pred, Y_test))
print(accuracy_score(Y_test, Y_pred))
