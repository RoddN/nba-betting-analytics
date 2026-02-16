#!/usr/bin/env python3
"""
Dashboard Streamlit per NBA Analytics
"""

import os
import json
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime

from ml_predictor import carica_modello, predici
from rag_engine import inizializza as init_rag, cerca, chiedi

# Path assoluti
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, '../data')
CACHE_FILE = os.path.join(DATA_DIR, 'events_cache.json')
DATA_FILE = os.path.join(DATA_DIR, 'nba_2008-2025_betting.csv')

st.set_page_config(page_title="NBA Analytics", page_icon="🏀", layout="wide")

SQUADRE = ['ATL', 'BOS', 'BKN', 'CHA', 'CHI', 'CLE', 'DAL', 'DEN',
           'DET', 'GSW', 'HOU', 'IND', 'LAC', 'LAL', 'MEM', 'MIA',
           'MIL', 'MIN', 'NOP', 'NYK', 'OKC', 'ORL', 'PHI', 'PHX',
           'POR', 'SAC', 'SAS', 'TOR', 'UTA', 'WAS']


@st.cache_resource
def carica_ml():
    try:
        return carica_modello()
    except:
        return None, None


@st.cache_resource  
def carica_rag():
    try:
        init_rag()
        return True
    except:
        return False


def tab_streaming():
    """Tab per visualizzare eventi Kafka in tempo reale"""
    st.header("📡 Live Streaming")
    
    col1, col2 = st.columns([3, 1])
    with col2:
        if st.button("🔄 Aggiorna"):
            st.rerun()
    
    with col1:
        st.info("Eventi Kafka salvati dal consumer. Esegui `python spark_consumer.py` per popolare.")
    
    if os.path.exists(CACHE_FILE):
        try:
            with open(CACHE_FILE, 'r') as f:
                data = json.load(f)
            
            # Supporta sia formato lista che dict con chiave 'events'
            if isinstance(data, dict) and 'events' in data:
                eventi = data['events']
            elif isinstance(data, list):
                eventi = data
            else:
                eventi = []
            
            if len(eventi) > 0:
                st.success(f"📊 {len(eventi)} eventi in cache")
                
                # Mostra ultimi 10 eventi
                st.subheader("Ultimi Eventi")
                for e in reversed(eventi[-10:]):
                    tipo = e.get('event_type', 'unknown')
                    home = e.get('home_team', '?')
                    away = e.get('away_team', '?')
                    
                    if tipo == 'game_end':
                        winner = e.get('winner', '?')
                        margin = e.get('margin', 0)
                        st.markdown(f"🏁 **FINE** | {home} {e.get('home_score',0)} - {e.get('away_score',0)} {away} | 🏆 {winner} (+{margin})")
                    elif tipo == 'quarter_end':
                        q = e.get('quarter', '?')
                        st.markdown(f"⏱️ Q{q} | {home} {e.get('home_score',0)} - {e.get('away_score',0)} {away}")
                    elif tipo == 'game_start':
                        st.markdown(f"🏀 **INIZIO** | {home} vs {away}")
                    else:
                        st.markdown(f"📢 {tipo} | {home} vs {away}")
            else:
                st.warning("Cache vuota. Esegui producer.py e spark_consumer.py")
        except Exception as e:
            st.error(f"Errore lettura cache: {e}")
    else:
        st.warning("⚠️ File cache non trovato. Esegui `python spark_consumer.py` per salvare gli eventi.")
        st.code("python producer.py      # Invia eventi a Kafka\npython spark_consumer.py  # Salva eventi nella cache", language="bash")


def tab_predizioni():
    """Tab per predizioni ML"""
    st.header("🤖 Predizioni ML")
    
    modello, scaler = carica_ml()
    if modello is None:
        st.error("❌ Modello non caricato")
        st.code("python3 ml_predictor.py --train", language="bash")
        return
    
    st.success("✅ Modello ML caricato!")
    
    col1, col2 = st.columns(2)
    with col1:
        home = st.selectbox("🏠 Squadra Casa", SQUADRE, index=SQUADRE.index('LAL'))
        away = st.selectbox("✈️ Squadra Ospite", SQUADRE, index=SQUADRE.index('BOS'))
    with col2:
        spread = st.slider("📊 Spread", -15.0, 15.0, 0.0, help="Differenza punti attesa (+ = favorita casa)")
    
    if st.button("🔮 Predici Vincitore", type="primary"):
        if home == away:
            st.error("Seleziona squadre diverse!")
        else:
            with st.spinner("Calcolo predizione..."):
                # Usa sempre le statistiche automatiche (live o training)
                pred = predici(modello, scaler, home, away, spread, 220, None, None)
            
            # Mostra statistiche squadre
            st.subheader("📊 Statistiche Squadre")
            c1, c2 = st.columns(2)
            with c1:
                st.metric(f"🏠 {home} Win Rate", f"{pred.get('home_win_rate', 0.5):.1%}")
                st.caption(f"Media punti: {pred.get('home_avg_pts', 105):.1f}")
            with c2:
                st.metric(f"✈️ {away} Win Rate", f"{pred.get('away_win_rate', 0.5):.1%}")
                st.caption(f"Media punti: {pred.get('away_avg_pts', 105):.1f}")
            
            st.subheader("🔮 Predizione")
            c1, c2, c3 = st.columns(3)
            c1.metric(f"Prob. {home}", f"{pred['prob_vittoria_casa']:.1%}")
            c2.metric(f"Prob. {away}", f"{pred['prob_vittoria_trasferta']:.1%}")
            c3.metric("Confidenza", f"{pred['confidenza']}%")
            
            vincitore = pred['vincitore_predetto']
            
            # Mostra fonte dati
            if pred.get('home_win_rate', 0) > 0:
                source = "live" if os.path.exists(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../data/live_stats.json')) else "training"
                st.caption(f"📊 Statistiche da: **{source}** layer")
            
            if pred['prob_vittoria_casa'] > pred['prob_vittoria_trasferta']:
                st.success(f"**🏆 Vincitore Predetto: {vincitore}** (in casa)")
            else:
                st.success(f"**🏆 Vincitore Predetto: {vincitore}** (in trasferta)")


def tab_rag():
    """Tab per ricerca semantica e Q&A"""
    st.header("🔍 Ricerca Semantica RAG")
    
    rag_ok = carica_rag()
    if not rag_ok:
        st.error("❌ RAG non inizializzato")
        st.code("python3 rag_engine.py --index", language="bash")
        return
    
    st.success("✅ RAG Engine pronto! (23.118 partite indicizzate)")
    
    # Due modalità: Ricerca e Domanda
    mode = st.radio("Modalità:", ["🔎 Cerca Partite", "💬 Fai una Domanda"], horizontal=True)
    
    if mode == "🔎 Cerca Partite":
        query = st.text_input("Cerca partite:", placeholder="es. Lakers vs Celtics, vittorie Warriors...")
        
        if st.button("🔎 Cerca", type="primary"):
            if query:
                with st.spinner("Ricerca in corso..."):
                    risultati = cerca(query, n_risultati=5)
                
                if risultati:
                    st.subheader(f"Trovate {len(risultati)} partite simili")
                    for i, r in enumerate(risultati, 1):
                        with st.expander(f"#{i} - Similarità: {r['similarita']:.0%}"):
                            st.write(r['documento'])
                            st.json(r['metadata'])
                else:
                    st.warning("Nessun risultato trovato")
            else:
                st.warning("Inserisci una query")
    
    else:  # Modalità Domanda
        st.info("💡 Fai domande sui dati NBA. Richiede GROQ_API_KEY nel file .env")
        domanda = st.text_area("La tua domanda:", placeholder="es. Quante partite hanno vinto i Lakers contro i Celtics?")
        
        if st.button("💬 Chiedi", type="primary"):
            if domanda:
                with st.spinner("Elaborazione risposta..."):
                    risposta = chiedi(domanda, n_contesto=5)
                
                st.subheader("📝 Risposta")
                st.write(risposta['risposta'])
                
                if risposta.get('fonti'):
                    with st.expander("📚 Fonti utilizzate"):
                        for f in risposta['fonti']:
                            st.write(f"- {f['documento']}")
            else:
                st.warning("Inserisci una domanda")


def tab_analytics():
    """Tab per analytics e statistiche - combina dati storici e live"""
    st.header("📊 Analytics")
    
    # Path per live stats
    LIVE_STATS_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../data/live_stats.json')
    
    col1, col2 = st.columns([3, 1])
    with col2:
        if st.button("🔄 Aggiorna", key="refresh_analytics"):
            st.rerun()
    
    # === SEZIONE 1: STATS LIVE (Speed Layer) ===
    st.subheader("⚡ Statistiche Real-Time")
    
    if os.path.exists(LIVE_STATS_FILE):
        try:
            with open(LIVE_STATS_FILE, 'r') as f:
                live_data = json.load(f)
            
            teams = live_data.get('teams', {})
            games_processed = live_data.get('games_processed', 0)
            last_update = live_data.get('last_update', 'N/A')
            
            if teams:
                col1, col2, col3 = st.columns(3)
                col1.metric("🎮 Partite Processate", games_processed)
                col2.metric("👥 Squadre Tracciate", len(teams))
                col3.metric("🕐 Ultimo Aggiornamento", last_update[:19] if last_update != 'N/A' else 'N/A')
                
                # Top squadre per win rate LIVE
                st.subheader("🏆 Top 10 Win Rate (Live)")
                
                sorted_teams = sorted(
                    [(k, v) for k, v in teams.items() if v.get('games', 0) > 0],
                    key=lambda x: x[1].get('win_rate', 0),
                    reverse=True
                )[:10]
                
                if sorted_teams:
                    team_names = [t[0] for t in sorted_teams]
                    win_rates = [t[1].get('win_rate', 0) * 100 for t in sorted_teams]
                    games = [t[1].get('games', 0) for t in sorted_teams]
                    
                    fig = go.Figure(data=[
                        go.Bar(x=team_names, y=win_rates, text=[f"{wr:.1f}%" for wr in win_rates], textposition='auto')
                    ])
                    fig.update_layout(yaxis_title='Win Rate (%)', xaxis_title='Squadra')
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Tabella dettagliata
                    with st.expander("📋 Dettagli Squadre"):
                        data = []
                        for team, stats in sorted_teams:
                            data.append({
                                'Squadra': team,
                                'Win Rate': f"{stats.get('win_rate', 0):.1%}",
                                'Vittorie': stats.get('wins', 0),
                                'Partite': stats.get('games', 0),
                                'Punti Medi': f"{stats.get('avg_pts', 0):.1f}"
                            })
                        st.dataframe(pd.DataFrame(data), use_container_width=True)
            else:
                st.info("⏳ In attesa di partite... Esegui producer.py e spark_consumer.py")
        except Exception as e:
            st.error(f"Errore lettura live stats: {e}")
    else:
        st.warning("⚠️ Stats live non disponibili. Avvia spark_consumer.py per iniziare a processare eventi.")
    
    st.markdown("---")
    
    # === SEZIONE 2: DATI STORICI (Batch Layer) ===
    st.subheader("📚 Dataset Storico")
    
    if os.path.exists(DATA_FILE):
        df = pd.read_csv(DATA_FILE)
        
        if 'score_home' in df.columns:
            df = df.rename(columns={
                'score_home': 'home_score',
                'score_away': 'away_score',
                'home': 'home_team',
                'away': 'away_team'
            })
        
        col1, col2, col3 = st.columns(3)
        col1.metric("📊 Partite Storiche", f"{len(df):,}")
        col2.metric("📅 Stagioni", df['season'].nunique() if 'season' in df.columns else "N/A")
        if 'home_score' in df.columns:
            col3.metric("🏀 Media Punti", f"{(df['home_score'].mean() + df['away_score'].mean())/2:.1f}")
        
        with st.expander("📋 Preview Dataset"):
            st.dataframe(df.head(15), use_container_width=True)
    else:
        st.error("Dataset storico non trovato")


def main():
    st.title("🏀 NBA Betting Analytics")
    
    tab1, tab2, tab3, tab4 = st.tabs(["📡 Stream", "🤖 Predizioni", "🔍 RAG", "📊 Analytics"])
    
    with tab1:
        tab_streaming()
    with tab2:
        tab_predizioni()
    with tab3:
        tab_rag()
    with tab4:
        tab_analytics()
    
    st.caption("Progetto InfoSys - NBA Betting Analytics")


if __name__ == '__main__':
    main()
