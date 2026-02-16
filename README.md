# NBA Betting Analytics Pipeline

Sistema di analytics in tempo reale per scommesse NBA con architettura streaming, machine learning e ricerca semantica RAG.

## Panoramica

Questo progetto implementa una pipeline completa per l'analisi di dati NBA betting:

```
┌──────────────┐    ┌─────────┐    ┌────────────────┐    ┌─────────────┐
│   Producer   │───▶│  Kafka  │───▶│ Spark Consumer │───▶│   Parquet   │
│  (eventi)    │    │  Topic  │    │  (streaming)   │    │   (storage) │
└──────────────┘    └─────────┘    └────────────────┘    └──────┬──────┘
                                                                │
                         ┌──────────────────────────────────────┘
                         ▼
┌─────────────────────────────────────────────────────────────────────┐
│                         Dashboard Streamlit                          │
│  ┌───────────┐  ┌──────────────┐  ┌───────────┐  ┌───────────────┐  │
│  │Live Stream│  │ Predizioni ML│  │ RAG Search│  │   Analytics   │  │
│  └───────────┘  └──────────────┘  └───────────┘  └───────────────┘  │
└─────────────────────────────────────────────────────────────────────┘
```

## Caratteristiche

- **Streaming Real-time**: Eventi NBA via Apache Kafka (16 eventi/partita)
- **Processing Distribuito**: Apache Spark Streaming per aggregazioni
- **Machine Learning**: RandomForest per predizioni probabilità vittoria
- **RAG Engine**: Ricerca semantica su 23.000+ partite storiche con ChromaDB + Groq LLM
- **Dashboard Interattiva**: Streamlit con auto-refresh e visualizzazioni Plotly

## Tecnologie

| Componente | Tecnologia |
|------------|------------|
| Message Broker | Apache Kafka |
| Stream Processing | Apache Spark |
| ML Model | Scikit-learn (RandomForest) |
| Vector DB | ChromaDB |
| Embeddings | Sentence-Transformers |
| LLM | Groq (Llama 3.3) |
| Dashboard | Streamlit + Plotly |
| Container | Docker Compose |

## Quick Start

```bash
# 1. Clone e setup
git clone https://github.com/YOUR_USERNAME/nba-betting-analytics.git
cd nba-betting-analytics

# 2. Virtual environment
python3 -m venv venv
source venv/bin/activate
pip install -r nba-analytics/requirements.txt

# 3. Avvia Kafka
cd nba-analytics
docker compose up -d

# 4. Avvia Dashboard
streamlit run dashboard.py
```

Apri http://localhost:8501

## Struttura Progetto

```
├── nba-analytics/
│   ├── producer.py           # Kafka producer - genera eventi NBA
│   ├── spark_consumer.py     # Spark Streaming consumer
│   ├── ml_predictor.py       # Modello ML predizioni
│   ├── rag_engine.py         # RAG engine con ChromaDB
│   ├── dashboard.py          # Dashboard Streamlit
│   ├── docker-compose.yml    # Kafka + Zookeeper
│   └── requirements.txt      # Dipendenze Python
├── data/
│   └── nba_2008-2025_betting.csv  # Dataset storico
└── models/
    └── nba_rf_model.pkl      # Modello ML salvato
```

## Utilizzo Avanzato

### Modalità Streaming Continuo

```bash
# Producer che genera partite all'infinito
python producer.py --continuous --limit 5 --interval 10

# In un altro terminale: Spark consumer
python spark_consumer.py
```

### Configurazione RAG con Groq

```bash
# Crea file .env
echo "GROQ_API_KEY=your_api_key_here" > .env
```

## Dataset

Il sistema utilizza dati NBA dal 2008 al 2025:
- 23.119 partite storiche
- Punteggi, spread, totali, quote moneyline
- Generazione sintetica se CSV non disponibile

## Licenza

MIT License
