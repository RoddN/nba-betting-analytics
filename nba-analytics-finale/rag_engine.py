#!/usr/bin/env python3
"""
RAG Engine per ricerca semantica partite NBA
"""

import os
import pandas as pd
import numpy as np
from dotenv import load_dotenv

import chromadb
from sentence_transformers import SentenceTransformer

load_dotenv()

# Configurazione - path relativi allo script
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CHROMADB_DIR = os.path.join(SCRIPT_DIR, '../data/chromadb')
DATA_FILE = os.path.join(SCRIPT_DIR, '../data/nba_2008-2025_betting.csv')
COLLECTION_NAME = 'partite_nba'
EMBEDDING_MODEL = 'all-MiniLM-L6-v2'

NOMI_SQUADRE = {
    'ATL': 'Atlanta Hawks', 'BOS': 'Boston Celtics', 'BKN': 'Brooklyn Nets',
    'CHA': 'Charlotte Hornets', 'CHI': 'Chicago Bulls', 'CLE': 'Cleveland Cavaliers',
    'DAL': 'Dallas Mavericks', 'DEN': 'Denver Nuggets', 'DET': 'Detroit Pistons',
    'GSW': 'Golden State Warriors', 'HOU': 'Houston Rockets', 'IND': 'Indiana Pacers',
    'LAC': 'Los Angeles Clippers', 'LAL': 'Los Angeles Lakers', 'MEM': 'Memphis Grizzlies',
    'MIA': 'Miami Heat', 'MIL': 'Milwaukee Bucks', 'MIN': 'Minnesota Timberwolves',
    'NOP': 'New Orleans Pelicans', 'NYK': 'New York Knicks', 'OKC': 'Oklahoma City Thunder',
    'ORL': 'Orlando Magic', 'PHI': 'Philadelphia 76ers', 'PHX': 'Phoenix Suns',
    'POR': 'Portland Trail Blazers', 'SAC': 'Sacramento Kings', 'SAS': 'San Antonio Spurs',
    'TOR': 'Toronto Raptors', 'UTA': 'Utah Jazz', 'WAS': 'Washington Wizards'
}

# Variabili globali
_embedder = None
_client = None
_collection = None


def inizializza():
    """Inizializza ChromaDB e modello embeddings"""
    global _embedder, _client, _collection
    
    print("Inizializzazione RAG Engine...")
    os.makedirs(CHROMADB_DIR, exist_ok=True)
    
    print(f"  Caricamento modello: {EMBEDDING_MODEL}")
    _embedder = SentenceTransformer(EMBEDDING_MODEL)
    
    print(f"  Inizializzazione ChromaDB")
    _client = chromadb.PersistentClient(path=CHROMADB_DIR)
    _collection = _client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"}
    )
    
    print(f"  Partite indicizzate: {_collection.count()}")
    return True


def crea_descrizione(partita):
    """Crea descrizione testuale"""
    home = partita['home_team']
    away = partita['away_team']
    home_name = NOMI_SQUADRE.get(home, home)
    away_name = NOMI_SQUADRE.get(away, away)
    
    home_score = int(partita['home_score'])
    away_score = int(partita['away_score'])
    vincitore = home if home_score > away_score else away
    margine = abs(home_score - away_score)
    
    return f"{home_name} vs {away_name}. Risultato: {home} {home_score} - {away} {away_score}. Vittoria {vincitore} di {margine} punti."


def indicizza_partite(forza=False):
    """Indicizza partite nel database"""
    global _collection
    
    if _collection is None:
        inizializza()
    
    if _collection.count() > 0 and not forza:
        print(f"Database già popolato ({_collection.count()} partite)")
        return _collection.count()
    
    # Carica dati
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
    except FileNotFoundError:
        print("File non trovato!")
        return 0
    
    # Indicizza in batch
    batch_size = 500
    totale = 0
    
    for i in range(0, len(df), batch_size):
        batch = df.iloc[i:i+batch_size]
        
        documenti = []
        ids = []
        metadati = []
        
        for j, (_, row) in enumerate(batch.iterrows()):
            idx = i + j
            doc = crea_descrizione(row)
            documenti.append(doc)
            ids.append(f"partita_{idx}")
            
            metadati.append({
                'home_team': str(row['home_team']),
                'away_team': str(row['away_team']),
                'home_score': int(row['home_score']),
                'away_score': int(row['away_score']),
                'winner': row['home_team'] if row['home_score'] > row['away_score'] else row['away_team']
            })
        
        embeddings = _embedder.encode(documenti).tolist()
        
        _collection.add(
            documents=documenti,
            embeddings=embeddings,
            metadatas=metadati,
            ids=ids
        )
        
        totale += len(documenti)
        print(f"  Indicizzate {totale}/{len(df)} partite")
    
    print(f"\nCompletato: {_collection.count()} partite")
    return _collection.count()


def cerca(query, n_risultati=5):
    """Cerca partite"""
    if _collection is None:
        inizializza()
    
    if _collection.count() == 0:
        print("Database vuoto!")
        return []
    
    query_embedding = _embedder.encode([query]).tolist()
    
    risultati = _collection.query(
        query_embeddings=query_embedding,
        n_results=n_risultati,
        include=['documents', 'metadatas', 'distances']
    )
    
    output = []
    for i in range(len(risultati['ids'][0])):
        output.append({
            'documento': risultati['documents'][0][i],
            'metadata': risultati['metadatas'][0][i],
            'similarita': round(1 - risultati['distances'][0][i], 3)
        })
    
    return output


def chiedi(domanda, n_contesto=3):
    """Risponde a una domanda"""
    risultati = cerca(domanda, n_risultati=n_contesto)
    
    if not risultati:
        return {"risposta": "Non ho trovato dati.", "fonti": []}
    
    contesto = "\n".join([r['documento'] for r in risultati])
    
    try:
        from groq import Groq
        groq_key = os.environ.get('GROQ_API_KEY')
        
        if groq_key:
            client = Groq(api_key=groq_key)
            risposta = client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[
                    {"role": "system", "content": "Sei un esperto NBA. Rispondi in italiano."},
                    {"role": "user", "content": f"Dati:\n{contesto}\n\nDomanda: {domanda}"}
                ],
                temperature=0.3,
                max_tokens=500
            )
            return {"risposta": risposta.choices[0].message.content, "fonti": risultati}
    except:
        pass
    
    return {"risposta": f"Trovate {len(risultati)} partite. Configura GROQ_API_KEY per risposte AI.", "fonti": risultati}


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='NBA RAG Engine')
    parser.add_argument('--index', action='store_true', help='Indicizza partite')
    parser.add_argument('--cerca', type=str, help='Cerca partite')
    args = parser.parse_args()
    
    inizializza()
    
    if args.index:
        indicizza_partite()
    elif args.cerca:
        print(f"\nRicerca: '{args.cerca}'\n")
        for i, r in enumerate(cerca(args.cerca), 1):
            print(f"{i}. [{r['similarita']}] {r['documento']}\n")
    else:
        print(f"Partite: {_collection.count()}")
        print("Usa --index o --cerca")


if __name__ == '__main__':
    main()
