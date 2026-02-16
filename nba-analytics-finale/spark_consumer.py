#!/usr/bin/env python3
"""
Consumer Spark Streaming per eventi NBA
Aggiorna statistiche live quando arrivano risultati partite
"""

import os
import json
from datetime import datetime
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, from_json, when, current_timestamp
from pyspark.sql.types import StructType, StructField, StringType, IntegerType, DoubleType

# Configurazione
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
KAFKA_SERVER = 'localhost:9092'
TOPIC = 'nba-games'
OUTPUT_DIR = os.path.join(SCRIPT_DIR, '../data/aggregates')
CHECKPOINT_DIR = os.path.join(SCRIPT_DIR, '../data/checkpoints')
CACHE_FILE = os.path.join(SCRIPT_DIR, '../data/events_cache.json')
LIVE_STATS_FILE = os.path.join(SCRIPT_DIR, '../data/live_stats.json')
TRAINING_STATS_FILE = os.path.join(SCRIPT_DIR, '../models/team_stats.pkl')

# Schema eventi
SCHEMA_EVENTO = StructType([
    StructField("event_type", StringType()),
    StructField("game_id", StringType()),
    StructField("timestamp", StringType()),
    StructField("home_team", StringType()),
    StructField("away_team", StringType()),
    StructField("home_score", IntegerType()),
    StructField("away_score", IntegerType()),
    StructField("quarter", IntegerType()),
    StructField("winner", StringType()),
    StructField("margin", IntegerType()),
    StructField("spread", DoubleType()),
    StructField("total", DoubleType())
])


def carica_stats_iniziali():
    """Carica stats iniziali da training o crea vuote"""
    # Prima prova a caricare live stats esistenti
    if os.path.exists(LIVE_STATS_FILE):
        try:
            with open(LIVE_STATS_FILE, 'r') as f:
                data = json.load(f)
            print(f"  Stats live caricate: {len(data.get('teams', {}))} squadre")
            return data
        except:
            pass
    
    # Altrimenti inizializza da stats training
    try:
        import joblib
        training_stats = joblib.load(TRAINING_STATS_FILE)
        
        # Converti in formato live
        teams = {}
        for team, stats in training_stats.items():
            teams[team.upper()] = {
                'wins': int(stats['win_rate'] * stats['total_games']),
                'games': stats['total_games'],
                'points': [],
                'avg_pts': stats.get('avg_pts', 105)
            }
        
        live_data = {
            'teams': teams,
            'games_processed': 0,
            'last_update': datetime.now().isoformat()
        }
        
        print(f"  Stats inizializzate da training: {len(teams)} squadre")
        return live_data
    except:
        print("  Inizializzazione stats vuote")
        return {'teams': {}, 'games_processed': 0, 'last_update': None}


def aggiorna_live_stats(batch_df, batch_id):
    """Aggiorna statistiche live quando arrivano game_end events"""
    if batch_df.isEmpty():
        return
    
    # Filtra solo game_end
    game_ends = [row.asDict() for row in batch_df.filter(col("event_type") == "game_end").collect()]
    
    if not game_ends:
        return
    
    # Carica stats correnti
    if os.path.exists(LIVE_STATS_FILE):
        try:
            with open(LIVE_STATS_FILE, 'r') as f:
                live_data = json.load(f)
        except:
            live_data = carica_stats_iniziali()
    else:
        live_data = carica_stats_iniziali()
    
    teams = live_data.get('teams', {})
    games_count = live_data.get('games_processed', 0)
    
    # Processa ogni game_end
    for game in game_ends:
        home = game.get('home_team', '').upper()
        away = game.get('away_team', '').upper()
        home_score = game.get('home_score', 0) or 0
        away_score = game.get('away_score', 0) or 0
        winner = game.get('winner', '').upper()
        
        if not home or not away:
            continue
        
        # Inizializza squadre se non esistono
        if home not in teams:
            teams[home] = {'wins': 0, 'games': 0, 'points': [], 'avg_pts': 105}
        if away not in teams:
            teams[away] = {'wins': 0, 'games': 0, 'points': [], 'avg_pts': 105}
        
        # Aggiorna statistiche
        teams[home]['games'] += 1
        teams[away]['games'] += 1
        
        if winner == home:
            teams[home]['wins'] += 1
        else:
            teams[away]['wins'] += 1
        
        # Aggiorna punti (ultimi 20)
        teams[home]['points'].append(home_score)
        teams[away]['points'].append(away_score)
        teams[home]['points'] = teams[home]['points'][-20:]
        teams[away]['points'] = teams[away]['points'][-20:]
        
        # Calcola media punti
        teams[home]['avg_pts'] = sum(teams[home]['points']) / len(teams[home]['points'])
        teams[away]['avg_pts'] = sum(teams[away]['points']) / len(teams[away]['points'])
        
        games_count += 1
    
    # Calcola win rate per ogni squadra
    for team in teams:
        if teams[team]['games'] > 0:
            teams[team]['win_rate'] = teams[team]['wins'] / teams[team]['games']
        else:
            teams[team]['win_rate'] = 0.5
    
    # Salva
    live_data = {
        'teams': teams,
        'games_processed': games_count,
        'last_update': datetime.now().isoformat()
    }
    
    with open(LIVE_STATS_FILE, 'w') as f:
        json.dump(live_data, f, indent=2)
    
    print(f"  Stats aggiornate: {len(game_ends)} partite processate (totale: {games_count})")


def salva_cache(batch_df, batch_id):
    """Callback per salvare eventi nella cache JSON"""
    if batch_df.isEmpty():
        return
    
    nuovi_eventi = [row.asDict() for row in batch_df.collect()]
    
    # Carica cache esistente
    eventi = []
    if os.path.exists(CACHE_FILE):
        try:
            with open(CACHE_FILE, 'r') as f:
                data = json.load(f)
            if isinstance(data, dict) and 'events' in data:
                eventi = data['events']
            elif isinstance(data, list):
                eventi = data
        except:
            eventi = []
    
    # Aggiungi nuovi eventi (max 100)
    eventi.extend(nuovi_eventi)
    eventi = eventi[-100:]
    
    with open(CACHE_FILE, 'w') as f:
        json.dump(eventi, f, indent=2)
    
    print(f"  Cache: {len(eventi)} eventi")


def crea_spark():
    """Crea sessione Spark"""
    print("Creazione sessione Spark...")
    
    spark = (SparkSession.builder
        .appName("NBA-Consumer")
        .master("local[*]")
        .config("spark.jars.packages", "org.apache.spark:spark-sql-kafka-0-10_2.12:3.5.0")
        .config("spark.sql.shuffle.partitions", "4")
        .getOrCreate()
    )
    
    spark.sparkContext.setLogLevel("WARN")
    print(f"Spark {spark.version} avviato")
    return spark


def leggi_stream_kafka(spark):
    """Legge stream da Kafka"""
    print(f"Connessione a Kafka: {TOPIC}")
    
    stream_raw = (spark.readStream
        .format("kafka")
        .option("kafka.bootstrap.servers", KAFKA_SERVER)
        .option("subscribe", TOPIC)
        .option("startingOffsets", "latest")
        .load()
    )
    
    stream = (stream_raw
        .selectExpr("CAST(value AS STRING) as json")
        .select(from_json(col("json"), SCHEMA_EVENTO).alias("data"))
        .select("data.*")
    )
    
    return stream


def main():
    """Main"""
    print("=== NBA Spark Consumer (Live Stats) ===\n")
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    
    # Inizializza stats
    print("Inizializzazione statistiche...")
    live_data = carica_stats_iniziali()
    with open(LIVE_STATS_FILE, 'w') as f:
        json.dump(live_data, f, indent=2)
    
    spark = crea_spark()
    stream = leggi_stream_kafka(spark)
    
    print("\nAvvio elaborazioni...")
    
    # Query per salvare cache eventi
    display = stream.select("event_type", "game_id", "home_team", "away_team", 
                           "home_score", "away_score", "winner", "margin", "quarter")
    
    query_cache = (display.writeStream
        .outputMode("append")
        .foreachBatch(salva_cache)
        .trigger(processingTime="5 seconds")
        .start()
    )
    
    # Query per aggiornare stats live (solo game_end)
    query_stats = (stream.writeStream
        .outputMode("append")
        .foreachBatch(aggiorna_live_stats)
        .trigger(processingTime="5 seconds")
        .start()
    )
    
    print(f"\nCache eventi: {CACHE_FILE}")
    print(f"Stats live: {LIVE_STATS_FILE}")
    print(f"Partite processate: {live_data.get('games_processed', 0)}")
    print("\nPremi Ctrl+C per terminare\n")
    
    try:
        spark.streams.awaitAnyTermination()
    except KeyboardInterrupt:
        print("\nArresto...")
        spark.stop()


if __name__ == '__main__':
    main()
