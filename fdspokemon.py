
import pandas as pd 
def feature_extraction(data:list[dict]) -> pd.DataFrame:
    
    """
    from the baseline (static features):
    a)  {Critical hits!
        p1_team max possible critical hit probability,
        p1_team avg critiical hit probability}

    from the timeline(dynamic features): 
    1. {p1_team and p2_team mean potential damage during the battle}
    2. {p1 number of missed moves and p2 number of missed moves per battle}
    3. {p1 and p2 team mean hp during the 30 rounds}
    4. {p1_team effective damage dealt from moves and/or caused by effects, and number of induced change of status}
    5. {p2_team effective damage dealt from moves and/or caused by effects, and number of induced change of status}
    6. {pokemon with a bad status for p1 and p2 in the last turn}
    """

    winning_features = []
    for battle in data:
        features = {} 
        
        p1_team = battle.get('p1_team_details', [])
        if p1_team:
        # a) --- Critical hits! ---
        # We calculate critical hits potential based on p1 pokemon speed stats
            team_speeds = [s.get('base_spe', 0) for s in p1_team]
            # Gen 1 critical hits calculation: T = base_speed/2, critical hits occur if random(0,255) < T
            crit_rates = []
            for speed in team_speeds:
                T = speed/2
                crit_probability = min(T/256, 1.0)  # probability = T/256
                crit_rates.append(crit_probability)
            
            # critical hit feature:
            features['p1_team_max_crit_rate'] = max(crit_rates) if crit_rates else 0  # because a critical hit is usually a rare event!
            features['p1_taem_avg_crit_rate'] = sum(crit_rates)/len(team_speeds) if crit_rates else 0

        # --- DYNAMIC FEATURES --- 
        timeline = battle.get('battle_timeline', [])
        
        # initialize counters for damaging moves and missed moves:
        base_power1 = 0
        base_power2 = 0
        
        null_moves1 = 0
        null_moves2 = 0
        
        p1_names = []
        p2_names = []
        
        # 1. --- p1_team and p2_team mean potential damange during battles ---
        # get their base power from move_details 
        for i in range(len(timeline)):
            x = timeline[i].get('p1_move_details')
            if (x):
                base_power1+=(x.get("base_power", 0))
            else:
                base_power1+=0
            if (x is None):
                null_moves1 += 1
            else:
                null_moves1 += 0
            #
            x = timeline[i].get('p2_move_details')
            if (x):
                base_power2+=(x.get("base_power", 0))
            else:
                base_power2+=0
            if (x is None):
                null_moves2 += 1
            else:
                null_moves2 += 0

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

        #
        base_power1/=len(timeline)
        features['p1_mean_potential_damage'] = base_power1
        base_power2/=len(timeline)
        features['p2_mean_potential_damage'] = base_power2
        
        # 2. --- p1 number of missed moves and p2 number of missed moves per battle ---
        features['p1_n_missed_moves'] = null_moves1
        features['p2_n_missed_moves'] = null_moves2
        
        # to get their health points we create a new dictionary for player 1 and player 2
        p1hp = {}
        p2hp = {}

        for el in p1_names:
            p1hp[el] = 100 #starting health 
        for el in p2_names:
            p2hp[el] = 100 #starting health

        #
        for i in range(len(timeline)):
            x = timeline[i].get('p1_pokemon_state')
            if (x):
                p1hp[x.get("name")] = x.get("hp_pct")*100

            x = timeline[i].get('p2_pokemon_state')
            if (x):
                p2hp[x.get("name")] = x.get("hp_pct")*100

        # 3. --- p1 and p2 team mean hp for 30 rounds --- 
        vl1 = 0
        for el in p1hp.keys():
            vl1 += p1hp[el] 

        vl2 = 0
        for el in p2hp.keys():
            vl2 += p2hp[el] 

        vl1 = vl1 + (6-len(p1hp.keys()))*100    # each player starts with 6 pokemons 
        vl2 = vl2 + (6-len(p2hp.keys()))*100

        vl1/=6
        vl2/=6

        features['p1_r30_mean_hpt'] = vl1
        features['p2_r30_mean_hpt'] = vl2

        # initialize counters for effective moves or moves that actually landed:
        p1_damage_dealt = 0
        p1_change_of_status = 0

        p2_damage_dealt = 0
        p2_change_of_status = 0
        
        # 4. --- p1_team mean effetive damage dealt and change of status induced---
        for event in range(len(timeline)):
            p2_team_state = timeline[event].get('p2_pokemon_state')
            if p2_team_state:
                current_hp_pct = p2_team_state.get('hp_pct', float)
                if current_hp_pct < 1.0:
                    p1_damage_dealt += (1.0 - current_hp_pct)
                current_status = p2_team_state.get('status', '')
                if current_status != 'nostatus':
                    p1_change_of_status += 1

        features['p1_r30_effectively_registered_mean_damage'] = p1_damage_dealt/len(timeline)
        features['p1_change_of_status_induced'] = p1_change_of_status

        # 5. --- p2_team mean effetive damage dealt and change of status induced---
        for event in range(len(timeline)):
            p1_team_state = timeline[event].get('p1_pokemon_state')
            if p1_team_state:
                current_hp_pct = p1_team_state.get('hp_pct', float)
                if current_hp_pct < 1.0:
                    p2_damage_dealt += (1.0 - current_hp_pct)
                current_status = p1_team_state.get('status', '')
                if current_status != 'nostatus':
                    p2_change_of_status += 1

        features['p2_r30_effectively_registered_mean_damage'] = p2_damage_dealt/len(timeline) 
        features['p2_change_of_status_induced'] = p2_change_of_status        

        # 6. --- a pokemon with a bad status for p1 and p2 in the last turn ---
        if timeline:
            last_turn = timeline[-1]
            p1_status = last_turn['p1_pokemon_state']['status']
            p2_status = last_turn['p2_pokemon_state']['status']
            features['p1_bad_status_at_last'] = 1 if p1_status != 'nostatus' else 0
            features['p2_bad_status_at_last'] = 1 if p2_status != 'nostatus' else 0 

        # Battle id and target variable:
        features['battle_id'] = battle.get('battle_id')
        if 'player_won' in battle:
            features['player_won'] = int(battle['player_won'])
        
        winning_features.append(features)
    
    return pd.DataFrame(winning_features).fillna(0)
