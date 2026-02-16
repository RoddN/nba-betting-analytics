# NBA Betting Analytics

Progetto di Information Systems (A.A. 2024/2025).

Il sistema prende un dataset di 23.118 partite NBA reali (stagioni 2008–2025), le usa per allenare un modello Random Forest che predice i vincitori, e permette di fare domande in linguaggio naturale sui dati tramite RAG. In parallelo, un producer Kafka simula partite in tempo reale che vengono processate con Spark Structured Streaming.

Tutto gira su una dashboard Streamlit.

## Come è organizzato

Il progetto segue la Lambda Architecture:

- **Batch layer** → `ml_predictor.py` allena il Random Forest sull'intero dataset storico
- **Speed layer** → `producer.py` manda eventi Kafka, `spark_consumer.py` li processa con Spark e aggiorna le statistiche live
- **Serving layer** → `dashboard.py` unisce i due layer e mostra tutto all'utente

## Stack

- **Kafka** per lo streaming (con Zookeeper, tutto dockerizzato)
- **Spark Structured Streaming** come consumer
- **scikit-learn** per il Random Forest
- **ChromaDB** + **all-MiniLM-L6-v2** per la ricerca semantica
- **Groq** (Llama 3.3 70B) per le risposte in linguaggio naturale (opzionale, serve la API key)
- **Streamlit** + **Plotly** per la dashboard

## File del progetto

```
nba-analytics-finale/
├── producer.py              # genera eventi e li manda a Kafka
├── spark_consumer.py        # legge da Kafka, aggiorna stats live
├── ml_predictor.py          # training + predizione Random Forest
├── rag_engine.py            # indicizzazione e ricerca semantica
├── dashboard.py             # dashboard Streamlit (4 tab)
├── grafici_performance.py   # genera i grafici per la presentazione
├── docker-compose.yml       # Kafka + Zookeeper + Kafka UI
├── requirements.txt
└── .env                     # GROQ_API_KEY (opzionale)
```

A runtime vengono generati anche:

- `models/` → modello salvato (.pkl), scaler, statistiche squadre, metriche
- `data/` → live_stats.json, events_cache.json, cartella chromadb
- `grafici/` → i PNG dei grafici di performance

## Setup

Serve Python 3.8+ e Docker.

```bash
# virtual environment
python3 -m venv ../venv
source ../venv/bin/activate
pip install -r requirements.txt

# avvia Kafka e Zookeeper
docker-compose up -d
```

Per usare la parte RAG con risposte in linguaggio naturale, va creato un file `.env` con `GROQ_API_KEY=gsk_xxxxx`. Senza la chiave funziona comunque la ricerca semantica, solo non vengono generate le risposte.

## Come si usa

La prima volta va allenato il modello e indicizzato il database RAG:

```bash
python3 ml_predictor.py --train    # allena il Random Forest (~1 min)
python3 rag_engine.py --index      # indicizza 23K partite in ChromaDB (~30 sec)
```

Dopo si può avviare tutto:

```bash
python3 producer.py          # terminale 1 - genera partite
python3 spark_consumer.py    # terminale 2 - processa con Spark
streamlit run dashboard.py   # terminale 3 - apre la dashboard
```

La dashboard si apre su `localhost:8501`. Kafka UI è su `localhost:8080`.

Per testare una predizione veloce da terminale:

```bash
python3 ml_predictor.py --test --home GSW --away LAL
```
## La dashboard

Ha 4 tab:

- **📡 Stream** — mostra gli eventi Kafka (inizio partita, fine quarti, fine partita)
- **🤖 Predizioni** — si selezionano due squadre e il modello restituisce la probabilità di vittoria
- **🔍 RAG** — si scrive una domanda tipo "vittorie Lakers contro Celtics" e il sistema cerca nel database
- **Analytics** — confronto tra statistiche live (speed layer) e storiche (batch layer)

## Dipendenze

Tutto in `requirements.txt`: kafka-python, pyspark, scikit-learn, chromadb, sentence-transformers, groq, streamlit, plotly, pandas, numpy, python-dotenv.