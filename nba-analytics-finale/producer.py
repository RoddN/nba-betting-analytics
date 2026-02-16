#!/usr/bin/env python3
"""
Producer Kafka per eventi NBA
Genera eventi partite e li invia a Kafka in formato Avro
"""
import os
import io
import time
import random
import pandas as pd
import fastavro
from kafka import KafkaProducer
from datetime import datetime, timedelta

# Configurazione
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
KAFKA_SERVER = 'localhost:9092'
TOPIC = 'nba-games'
DATA_FILE = os.path.join(SCRIPT_DIR, '../data/nba_2008-2025_betting.csv')

# Schema Avro per gli eventi NBA
SCHEMA_AVRO = {
    "type": "record",
    "name": "NBAEvent",
    "fields": [
        {"name": "event_type", "type": "string"},
        {"name": "game_id", "type": "string"},
        {"name": "timestamp", "type": "string"},
        {"name": "home_team", "type": "string"},
        {"name": "away_team", "type": "string"},
        {"name": "home_score", "type": ["null", "int"], "default": None},
        {"name": "away_score", "type": ["null", "int"], "default": None},
        {"name": "quarter", "type": ["null", "int"], "default": None},
        {"name": "winner", "type": ["null", "string"], "default": None},
        {"name": "margin", "type": ["null", "int"], "default": None},
        {"name": "spread", "type": ["null", "double"], "default": None},
        {"name": "total", "type": ["null", "double"], "default": None}
    ]
}
PARSED_SCHEMA = fastavro.parse_schema(SCHEMA_AVRO)

SQUADRE = ['ATL', 'BOS', 'BKN', 'CHA', 'CHI', 'CLE', 'DAL', 'DEN',
           'DET', 'GSW', 'HOU', 'IND', 'LAC', 'LAL', 'MEM', 'MIA',
           'MIL', 'MIN', 'NOP', 'NYK', 'OKC', 'ORL', 'PHI', 'PHX',
           'POR', 'SAC', 'SAS', 'TOR', 'UTA', 'WAS']


def carica_dati():
    """Carica il dataset NBA dal CSV"""
    try:
        df = pd.read_csv(DATA_FILE)
        if 'score_home' in df.columns:
            df = df.rename(columns={
                'score_home': 'home_score',
                'score_away': 'away_score',
                'home': 'home_team', 
                'away': 'away_team'
            })
        print(f"Caricate {len(df)} partite")
        return df
    except FileNotFoundError:
        print("File non trovato, genero dati di esempio...")
        return genera_dati_random(100)


def genera_dati_random(n_partite):
    """Genera dati finti per test"""
    partite = []
    for i in range(n_partite):
        home, away = random.sample(SQUADRE, 2)
        partite.append({
            'home_team': home,
            'away_team': away,
            'home_score': random.randint(90, 130),
            'away_score': random.randint(90, 130),
            'spread': round(random.uniform(-10, 10), 1),
            'total': round(random.uniform(200, 240), 1)
        })
    return pd.DataFrame(partite)


def serializza_avro(evento):
    """Serializza un evento in formato Avro binario"""
    buf = io.BytesIO()
    fastavro.schemaless_writer(buf, PARSED_SCHEMA, evento)
    return buf.getvalue()


def simula_partita_live(partita, producer):
    """Simula una partita in tempo reale con eventi progressivi"""
    game_id = f"game_{datetime.now().strftime('%H%M%S')}_{random.randint(100, 999)}"
    
    home = partita['home_team']
    away = partita['away_team']
    home_final = int(partita['home_score'])
    away_final = int(partita['away_score'])
    
    # Evento inizio partita
    evento_start = {
        'event_type': 'game_start',
        'game_id': game_id,
        'home_team': home,
        'away_team': away,
        'home_score': None,
        'away_score': None,
        'quarter': None,
        'winner': None,
        'margin': None,
        'spread': float(partita.get('spread', 0)),
        'total': float(partita.get('total', 220)),
        'timestamp': datetime.now().isoformat()
    }
    producer.send(TOPIC, evento_start)
    print(f"  INIZIO: {home} vs {away}")
    time.sleep(0.3)
    
    # Simula i 4 quarti con punteggi progressivi
    home_score = 0
    away_score = 0
    
    for quarter in range(1, 5):
        # Aggiungi punti del quarto (proporzionali al punteggio finale)
        home_q = int(home_final * 0.25) + random.randint(-3, 3)
        away_q = int(away_final * 0.25) + random.randint(-3, 3)
        home_score += home_q
        away_score += away_q
        
        evento_quarter = {
            'event_type': 'quarter_end',
            'game_id': game_id,
            'quarter': quarter,
            'home_team': home,
            'away_team': away,
            'home_score': home_score,
            'away_score': away_score,
            'winner': None,
            'margin': None,
            'spread': None,
            'total': None,
            'timestamp': datetime.now().isoformat()
        }
        producer.send(TOPIC, evento_quarter)
        print(f"    Q{quarter}: {home} {home_score} - {away} {away_score}")
        time.sleep(0.2)
    
    # Evento fine partita con punteggi finali reali
    vincitore = home if home_final > away_final else away
    evento_end = {
        'event_type': 'game_end',
        'game_id': game_id,
        'home_team': home,
        'away_team': away,
        'home_score': home_final,
        'away_score': away_final,
        'quarter': None,
        'winner': vincitore,
        'margin': abs(home_final - away_final),
        'spread': None,
        'total': None,
        'timestamp': datetime.now().isoformat()
    }
    producer.send(TOPIC, evento_end)
    print(f"  FINE: {home} {home_final} - {away} {away_final} | Vince {vincitore}")
    
    return 6  # Numero eventi generati


def main():
    """Funzione principale - loop continuo"""
    print("=== NBA Kafka Producer (Avro) ===")
    print(f"Server: {KAFKA_SERVER}")
    print(f"Topic: {TOPIC}")
    print(f"Formato: Avro")
    print("\nModalita LIVE - Ctrl+C per fermare\n")
    
    # Carica dati
    df = carica_dati()
    
    # Connessione a Kafka
    print("Connessione a Kafka...")
    try:
        producer = KafkaProducer(
            bootstrap_servers=KAFKA_SERVER,
            value_serializer=serializza_avro
        )
        print("Connesso!\n")
    except Exception as e:
        print(f"Errore connessione: {e}")
        print("Assicurati che Kafka sia attivo (docker-compose up -d)")
        return
    
    # Loop continuo
    eventi_totali = 0
    partite_giocate = 0
    
    try:
        while True:
            # Seleziona una partita random dal dataset
            partita = df.sample(1).iloc[0]
            
            # Simula la partita in tempo reale
            eventi = simula_partita_live(partita, producer)
            eventi_totali += eventi
            partite_giocate += 1
            
            producer.flush()
            
            # Pausa tra le partite (simula tempo reale)
            pausa = random.uniform(2, 4)
            print(f"\n  Totale: {partite_giocate} partite, {eventi_totali} eventi")
            print(f"  Prossima partita tra {pausa:.1f}s...\n")
            time.sleep(pausa)
            
    except KeyboardInterrupt:
        print(f"\n\nArresto producer...")
        print(f"Statistiche finali:")
        print(f"   - Partite simulate: {partite_giocate}")
        print(f"   - Eventi totali: {eventi_totali}")
        producer.close()


if __name__ == '__main__':
    main()
