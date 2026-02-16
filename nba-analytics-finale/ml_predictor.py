#!/usr/bin/env python3
"""
Modello ML per predizioni NBA
Usa Random Forest per predire vittorie casa/trasferta
"""

import os
import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import cross_val_score

import json

# Configurazione - path relativi allo script
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_FILE = os.path.join(SCRIPT_DIR, '../models/modello_nba.pkl')
SCALER_FILE = os.path.join(SCRIPT_DIR, '../models/scaler_nba.pkl')
STATS_FILE = os.path.join(SCRIPT_DIR, '../models/team_stats.pkl')
LIVE_STATS_FILE = os.path.join(SCRIPT_DIR, '../data/live_stats.json')
DATA_FILE = os.path.join(SCRIPT_DIR, '../data/nba_2008-2025_betting.csv')

SQUADRE = ['ATL', 'BOS', 'BKN', 'CHA', 'CHI', 'CLE', 'DAL', 'DEN',
           'DET', 'GSW', 'HOU', 'IND', 'LAC', 'LAL', 'MEM', 'MIA',
           'MIL', 'MIN', 'NOP', 'NYK', 'OKC', 'ORL', 'PHI', 'PHX',
           'POR', 'SAC', 'SAS', 'TOR', 'UTA', 'WAS']

# Statistiche squadre (globale, calcolate durante training)
_team_stats = None


def carica_dati():
    """Carica il dataset NBA"""
    try:
        df = pd.read_csv(DATA_FILE)
        print(f"Caricate {len(df)} partite")
    except FileNotFoundError:
        print("File non trovato, genero dati esempio...")
        df = genera_dati_esempio(5000)
        df['vittoria_casa'] = (df['score_home'] > df['score_away']).astype(int)
        return df
    
    # Rinomina colonne per coerenza
    if 'score_home' in df.columns:
        df = df.rename(columns={
            'score_home': 'home_score',
            'score_away': 'away_score', 
            'home': 'home_team',
            'away': 'away_team'
        })
    
    # Crea target
    df['vittoria_casa'] = (df['home_score'] > df['away_score']).astype(int)
    
    return df


def genera_dati_esempio(n):
    """Genera dataset di esempio per training"""
    np.random.seed(42)
    dati = []
    
    for _ in range(n):
        home, away = np.random.choice(SQUADRE, 2, replace=False)
        home_score = np.random.randint(90, 130)
        away_score = np.random.randint(90, 130)
        
        dati.append({
            'home_team': home,
            'away_team': away,
            'home_score': home_score,
            'away_score': away_score,
            'spread': round(np.random.uniform(-10, 10), 1),
            'total': round(np.random.uniform(200, 240), 1)
        })
    
    return pd.DataFrame(dati)


def calcola_statistiche_squadre(df):
    """Calcola statistiche per tutte le squadre"""
    stats = {}
    
    for sq in df['home_team'].unique().tolist() + df['away_team'].unique().tolist():
        if sq not in stats:
            # Partite in casa
            home_games = df[df['home_team'] == sq]
            home_wins = (home_games['home_score'] > home_games['away_score']).sum()
            home_pts = home_games['home_score'].mean() if len(home_games) > 0 else 105
            
            # Partite fuori
            away_games = df[df['away_team'] == sq]
            away_wins = (away_games['away_score'] > away_games['home_score']).sum()
            away_pts = away_games['away_score'].mean() if len(away_games) > 0 else 105
            
            total_games = len(home_games) + len(away_games)
            total_wins = home_wins + away_wins
            
            stats[sq] = {
                'win_rate': total_wins / total_games if total_games > 0 else 0.5,
                'home_win_rate': home_wins / len(home_games) if len(home_games) > 0 else 0.5,
                'away_win_rate': away_wins / len(away_games) if len(away_games) > 0 else 0.5,
                'avg_pts': (home_pts + away_pts) / 2,
                'home_avg_pts': home_pts if not pd.isna(home_pts) else 105,
                'away_avg_pts': away_pts if not pd.isna(away_pts) else 105,
                'total_games': total_games
            }
    
    return stats


def calcola_features(df, stats=None):
    """Calcola le features per il modello"""
    print("Calcolo features...")
    
    if stats is None:
        # Calcola statistiche progressivamente
        running_stats = {}
        for sq in SQUADRE:
            running_stats[sq] = {'vittorie': 0, 'partite': 0, 'punti': []}
        
        features = []
        for _, row in df.iterrows():
            home = row['home_team']
            away = row['away_team']
            
            if home not in running_stats:
                running_stats[home] = {'vittorie': 0, 'partite': 0, 'punti': []}
            if away not in running_stats:
                running_stats[away] = {'vittorie': 0, 'partite': 0, 'punti': []}
            
            home_wr = running_stats[home]['vittorie'] / max(running_stats[home]['partite'], 1)
            away_wr = running_stats[away]['vittorie'] / max(running_stats[away]['partite'], 1)
            home_pts = np.mean(running_stats[home]['punti'][-10:]) if running_stats[home]['punti'] else 105
            away_pts = np.mean(running_stats[away]['punti'][-10:]) if running_stats[away]['punti'] else 105
            
            features.append({
                'home_win_rate': home_wr,
                'away_win_rate': away_wr,
                'home_avg_pts': home_pts,
                'away_avg_pts': away_pts,
                'win_rate_diff': home_wr - away_wr,
                'spread': row.get('spread', 0) if pd.notna(row.get('spread', 0)) else 0,
                'total': row.get('total', 220) if pd.notna(row.get('total', 220)) else 220
            })
            
            # Aggiorna statistiche
            vittoria_casa = row['home_score'] > row['away_score']
            running_stats[home]['partite'] += 1
            running_stats[away]['partite'] += 1
            running_stats[home]['vittorie'] += 1 if vittoria_casa else 0
            running_stats[away]['vittorie'] += 0 if vittoria_casa else 1
            running_stats[home]['punti'].append(row['home_score'])
            running_stats[away]['punti'].append(row['away_score'])
        
        return pd.DataFrame(features)
    else:
        # Usa statistiche pre-calcolate
        features = []
        for _, row in df.iterrows():
            home = row['home_team']
            away = row['away_team']
            
            home_stats = stats.get(home, {'win_rate': 0.5, 'home_avg_pts': 105})
            away_stats = stats.get(away, {'win_rate': 0.5, 'away_avg_pts': 105})
            
            features.append({
                'home_win_rate': home_stats.get('home_win_rate', home_stats['win_rate']),
                'away_win_rate': away_stats.get('away_win_rate', away_stats['win_rate']),
                'home_avg_pts': home_stats.get('home_avg_pts', 105),
                'away_avg_pts': away_stats.get('away_avg_pts', 105),
                'win_rate_diff': home_stats['win_rate'] - away_stats['win_rate'],
                'spread': row.get('spread', 0) if pd.notna(row.get('spread', 0)) else 0,
                'total': row.get('total', 220) if pd.notna(row.get('total', 220)) else 220
            })
        
        return pd.DataFrame(features)


def allena_modello():
    """Allena il modello Random Forest"""
    global _team_stats
    print("=== Training Modello NBA ===\n")
    
    # Carica dati
    df = carica_dati()
    
    # Calcola statistiche squadre (per predizioni future)
    _team_stats = calcola_statistiche_squadre(df)
    print(f"Statistiche calcolate per {len(_team_stats)} squadre")
    
    # Calcola features
    features_df = calcola_features(df)
    feature_names = list(features_df.columns)
    
    # Prepara X e y
    X = features_df.values
    y = df['vittoria_casa'].values
    
    # Split train/test (80/20)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    print(f"Training: {len(X_train)}, Test: {len(X_test)}")
    
    # Scala le features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    # Cross-validation (5-fold)
    print("\nCross-Validation (5-fold)...")
    modello_cv = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
    cv_scores = cross_val_score(modello_cv, X_train, y_train, cv=5, scoring='accuracy')
    print(f"   Accuracy per fold: {[f'{s:.4f}' for s in cv_scores]}")
    print(f"   Media: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
    
    # Allena Random Forest (modello finale su tutto il training set)
    print("\nTraining Random Forest...")
    modello = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        random_state=42
    )
    modello.fit(X_train, y_train)
    
    # Valuta su test set
    y_pred = modello.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"\nAccuracy: {accuracy:.2%}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['Vittoria Trasferta', 'Vittoria Casa']))
    
    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    print("Confusion Matrix:")
    print(f"   {'':>20} Pred Trasf.  Pred Casa")
    print(f"   {'Reale Trasferta':>20}    {cm[0][0]:>5}       {cm[0][1]:>5}")
    print(f"   {'Reale Casa':>20}    {cm[1][0]:>5}       {cm[1][1]:>5}")
    
    # Feature Importance
    print("\nFeature Importance:")
    importances = modello.feature_importances_
    sorted_idx = np.argsort(importances)[::-1]
    for i in sorted_idx:
        print(f"   {feature_names[i]:>20}: {importances[i]:.4f} {'█' * int(importances[i] * 40)}")
    
    # Mostra top squadre
    print("\nTop 5 squadre per win rate:")
    sorted_teams = sorted(_team_stats.items(), key=lambda x: x[1]['win_rate'], reverse=True)[:5]
    for team, stats in sorted_teams:
        print(f"   {team}: {stats['win_rate']:.1%} ({stats['total_games']} partite)")
    
    # Salva modello, scaler, statistiche e metriche
    os.makedirs(os.path.dirname(MODEL_FILE), exist_ok=True)
    joblib.dump(modello, MODEL_FILE)
    joblib.dump(scaler, SCALER_FILE)
    joblib.dump(_team_stats, STATS_FILE)
    
    # Salva metriche per grafici
    metriche = {
        'accuracy': accuracy,
        'cv_scores': cv_scores.tolist(),
        'cv_mean': cv_scores.mean(),
        'cv_std': cv_scores.std(),
        'confusion_matrix': cm.tolist(),
        'feature_names': feature_names,
        'feature_importances': importances.tolist(),
        'classification_report': classification_report(y_test, y_pred, target_names=['Vittoria Trasferta', 'Vittoria Casa'], output_dict=True),
        'y_test': y_test.tolist(),
        'y_pred': y_pred.tolist(),
        'y_proba': modello.predict_proba(X_test).tolist()
    }
    metriche_file = os.path.join(os.path.dirname(MODEL_FILE), 'metriche.pkl')
    joblib.dump(metriche, metriche_file)
    
    print(f"\nModello salvato in: {MODEL_FILE}")
    print(f"Metriche salvate in: {metriche_file}")
    
    return modello, scaler


def carica_modello():
    """Carica modello salvato"""
    global _team_stats
    try:
        modello = joblib.load(MODEL_FILE)
        scaler = joblib.load(SCALER_FILE)
        _team_stats = joblib.load(STATS_FILE)
        print("Modello caricato")
        return modello, scaler
    except FileNotFoundError:
        print("Modello non trovato, avvio training...")
        return allena_modello()


def get_team_stats(team):
    """
    Ottieni statistiche di una squadra.
    Prima prova da live_stats.json (speed layer), poi fallback su training stats.
    """
    global _team_stats
    team_upper = team.upper()
    team_lower = team.lower()
    
    default_stats = {'win_rate': 0.5, 'home_avg_pts': 105, 'away_avg_pts': 105, 
                     'home_win_rate': 0.5, 'away_win_rate': 0.5, 'avg_pts': 105}
    
    # 1. Prima prova da live_stats.json (dati in tempo reale)
    if os.path.exists(LIVE_STATS_FILE):
        try:
            with open(LIVE_STATS_FILE, 'r') as f:
                live_data = json.load(f)
            teams = live_data.get('teams', {})
            
            if team_upper in teams:
                t = teams[team_upper]
                return {
                    'win_rate': t.get('win_rate', 0.5),
                    'home_win_rate': t.get('win_rate', 0.5),
                    'away_win_rate': t.get('win_rate', 0.5),
                    'avg_pts': t.get('avg_pts', 105),
                    'home_avg_pts': t.get('avg_pts', 105),
                    'away_avg_pts': t.get('avg_pts', 105),
                    'games': t.get('games', 0),
                    'wins': t.get('wins', 0),
                    'source': 'live'
                }
        except:
            pass
    
    # 2. Fallback su training stats
    if _team_stats is None:
        try:
            _team_stats = joblib.load(STATS_FILE)
        except:
            return default_stats
    
    # Cerca con vari formati nome
    if team in _team_stats:
        stats = _team_stats[team]
        stats['source'] = 'training'
        return stats
    elif team_lower in _team_stats:
        stats = _team_stats[team_lower]
        stats['source'] = 'training'
        return stats
    elif team_upper in _team_stats:
        stats = _team_stats[team_upper]
        stats['source'] = 'training'
        return stats
    
    return default_stats


def predici(modello, scaler, home_team, away_team, spread=0, total=220, 
            home_wr=None, away_wr=None):
    """
    Fa una predizione per una partita.
    Se home_wr/away_wr non sono forniti, usa le statistiche storiche delle squadre.
    """
    
    # Ottieni statistiche reali delle squadre se non fornite
    home_stats = get_team_stats(home_team)
    away_stats = get_team_stats(away_team)
    
    # Usa win rate forniti dall'utente o quelli storici
    if home_wr is None:
        home_wr = home_stats.get('home_win_rate', home_stats.get('win_rate', 0.5))
    if away_wr is None:
        away_wr = away_stats.get('away_win_rate', away_stats.get('win_rate', 0.5))
    
    # Usa punti medi storici
    home_pts = home_stats.get('home_avg_pts', 105)
    away_pts = away_stats.get('away_avg_pts', 105)
    
    # Prepara features
    features = np.array([[
        home_wr,
        away_wr,
        home_pts,
        away_pts,
        home_wr - away_wr,
        spread,
        total
    ]])
    
    # Scala e predici
    features_scaled = scaler.transform(features)
    prob = modello.predict_proba(features_scaled)[0]
    
    return {
        'home_team': home_team,
        'away_team': away_team,
        'prob_vittoria_casa': round(prob[1], 3),
        'prob_vittoria_trasferta': round(prob[0], 3),
        'vincitore_predetto': home_team if prob[1] > 0.5 else away_team,
        'confidenza': round(max(prob) * 100, 1),
        'home_win_rate': round(home_wr, 3),
        'away_win_rate': round(away_wr, 3),
        'home_avg_pts': round(home_pts, 1),
        'away_avg_pts': round(away_pts, 1)
    }


def main():
    """Funzione principale"""
    import argparse
    
    parser = argparse.ArgumentParser(description='ML Predictor NBA')
    parser.add_argument('--train', action='store_true', help='Allena nuovo modello')
    parser.add_argument('--test', action='store_true', help='Test predizione')
    parser.add_argument('--home', type=str, default='LAL', help='Squadra casa')
    parser.add_argument('--away', type=str, default='BOS', help='Squadra ospite')
    args = parser.parse_args()
    
    if args.train:
        allena_modello()
    elif args.test:
        modello, scaler = carica_modello()
        
        print(f"\n=== Test Predizione: {args.home} vs {args.away} ===")
        pred = predici(modello, scaler, 
                      home_team=args.home, 
                      away_team=args.away,
                      spread=-3.5)
        
        print(f"\nStatistiche:")
        print(f"   {pred['home_team']}: {pred['home_win_rate']:.1%} win rate, {pred['home_avg_pts']:.1f} PPG")
        print(f"   {pred['away_team']}: {pred['away_win_rate']:.1%} win rate, {pred['away_avg_pts']:.1f} PPG")
        print(f"\nPredizione:")
        print(f"   Prob. {pred['home_team']}: {pred['prob_vittoria_casa']:.1%}")
        print(f"   Prob. {pred['away_team']}: {pred['prob_vittoria_trasferta']:.1%}")
        print(f"   Vincitore: {pred['vincitore_predetto']} (confidenza: {pred['confidenza']}%)")
    else:
        modello, scaler = carica_modello()
        print("\nUsa --train per allenare o --test per testare")
        print("Esempio: python ml_predictor.py --test --home GSW --away LAL")


if __name__ == '__main__':
    main()
