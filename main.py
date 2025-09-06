import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import requests

# URL de l'API FastAPI
API_URL = "http://127.0.0.1:8000/predict_csv"

# Config
st.set_page_config(page_title="Analyse Billets", page_icon="💰", layout="wide")

# CSS Design Moderne 2024
st.markdown("""
    <style>
    /* Import des polices modernes */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    /* Fond moderne avec texture subtile */
    .stApp {
        background: linear-gradient(135deg, #0f0f23 0%, #1a1a2e 50%, #16213e 100%);
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        color: #e8eaed;
    }

    /* Header élégant avec effet de profondeur */
    .header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2.5rem;
        border-radius: 20px;
        text-align: center;
        color: white;
        margin-bottom: 2rem;
        box-shadow: 0 20px 40px rgba(102, 126, 234, 0.15);
        position: relative;
        overflow: hidden;
    }
    
    .header::before {
        content: '';
        position: absolute;
        top: 0;
        left: -100%;
        width: 100%;
        height: 100%;
        background: linear-gradient(90deg, transparent, rgba(255,255,255,0.1), transparent);
        animation: shine 3s infinite;
    }
    
    @keyframes shine {
        0% { left: -100%; }
        100% { left: 100%; }
    }

    /* Cartes avec effet moderne */
    .glass-card {
        background: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 16px;
        padding: 1.5rem;
        backdrop-filter: blur(20px);
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.2);
        margin-bottom: 1.5rem;
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    
    .glass-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 12px 40px rgba(0, 0, 0, 0.3);
    }

    /* Zone upload moderne */
    .upload-box {
        border: 2px dashed #4f46e5;
        border-radius: 16px;
        padding: 2rem;
        text-align: center;
        transition: all 0.3s ease;
        color: #e8eaed;
        background: rgba(79, 70, 229, 0.02);
    }
    
    .upload-box:hover {
        background: rgba(79, 70, 229, 0.08);
        border-color: #6366f1;
        transform: translateY(-2px);
    }

    /* Carte graphique moderne */
    .chart-box {
        background: linear-gradient(145deg, #1f2937, #111827);
        border: 1px solid rgba(79, 70, 229, 0.3);
        border-radius: 20px;
        padding: 2rem;
        color: #e8eaed;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.3);
        margin-top: 1.5rem;
        position: relative;
        overflow: hidden;
    }
    
    .chart-box::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 1px;
        background: linear-gradient(90deg, transparent, #4f46e5, transparent);
    }

    /* Métriques modernes */
    .metric-card {
        background: linear-gradient(135deg, rgba(79, 70, 229, 0.1), rgba(124, 58, 237, 0.05));
        border: 1px solid rgba(79, 70, 229, 0.2);
        border-radius: 16px;
        padding: 1.5rem;
        text-align: center;
        backdrop-filter: blur(10px);
        transition: all 0.3s ease;
        color: #e8eaed;
    }
    
    .metric-card:hover {
        transform: scale(1.02);
        border-color: rgba(79, 70, 229, 0.4);
    }
    
    .metric-card h3 {
        margin: 0 0 0.5rem 0;
        font-size: 0.875rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: #9ca3af;
    }
    
    .metric-card p {
        margin: 0;
        font-size: 2rem;
        font-weight: 700;
        color: #4f46e5;
    }

    /* Boutons modernes */
    .stButton > button {
        background: linear-gradient(135deg, #4f46e5, #7c3aed) !important;
        color: white !important;
        border: none !important;
        border-radius: 12px !important;
        padding: 0.75rem 2rem !important;
        font-weight: 600 !important;
        transition: all 0.2s ease !important;
        box-shadow: 0 4px 14px rgba(79, 70, 229, 0.3) !important;
    }
    
    .stButton > button:hover {
        transform: translateY(-1px) !important;
        box-shadow: 0 6px 20px rgba(79, 70, 229, 0.4) !important;
    }

    </style>
""", unsafe_allow_html=True)

# Header
st.markdown("""
<div class='header'>
    <h1 style='margin: 0; font-size: 2.5rem; font-weight: 700;'> Détection de Faux Billets</h1>
</div>
""", unsafe_allow_html=True)

# Layout
col1, col2 = st.columns([1,2])

with col1:
    st.markdown("<div class='glass-card'><h3>Importer vos données</h3>", unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Déposez un CSV ici", type=["csv"])
    chart_type = st.selectbox("Type de graphique", ["Camembert", "Barres"])
    st.markdown("</div>", unsafe_allow_html=True)

with col2:
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)

        st.markdown("<div class='glass-card'><h3> Aperçu</h3></div>", unsafe_allow_html=True)
        st.dataframe(df.head())

        if st.button("Lancer les prédictions via FastAPI"):
            try:
                # Envoyer fichier à FastAPI
                files = {"file": (uploaded_file.name, uploaded_file.getvalue(), "text/csv")}
                response = requests.post(API_URL, files=files)

                if response.status_code == 200:
                    result = response.json()
                    
                    # Vérifier si l'API a retourné une erreur
                    if "error" in result:
                        st.error(f"Erreur de l'API: {result['error']}")
                    else:
                        # Ajouter les prédictions au DataFrame
                        df['Prédiction'] = result["predictions"]
                        
                        # Ajouter les résultats textuels si disponibles
                        if "resultats" in result:
                            df['Résultat'] = result["resultats"]

                        # Calcul des statistiques
                        vrais = int((df['Prédiction'] == 1).sum())
                        faux = int((df['Prédiction'] == 0).sum())
                        total = len(df)

                        # Affichage des métriques avec nouveau style
                        c1, c2, c3 = st.columns(3)
                        with c1: 
                            st.markdown(f"<div class='metric-card'><h3>✅ Vrais billets</h3><p style='color: #10b981;'>{vrais}</p></div>", unsafe_allow_html=True)
                        with c2: 
                            st.markdown(f"<div class='metric-card'><h3>❌ Faux billets</h3><p style='color: #ef4444;'>{faux}</p></div>", unsafe_allow_html=True)
                        with c3: 
                            pourcentage = round((vrais / total) * 100, 1) if total > 0 else 0
                            st.markdown(f"<div class='metric-card'><h3>📦 Total</h3><p>{total}</p></div>", unsafe_allow_html=True)

                        # Graphique avec nouvelles couleurs modernes
                        st.markdown("<div class='chart-box'><h3 style='margin-top: 0;'>📈 Résultats</h3>", unsafe_allow_html=True)
                        
                        # Configuration matplotlib avec fond moderne
                        fig, ax = plt.subplots(facecolor="#1f2937", figsize=(10, 6))
                        fig.patch.set_facecolor('#1f2937')
                        
                        counts = df['Prédiction'].value_counts()

                        if chart_type == "Camembert":
                            # Nouvelles couleurs modernes
                            couleurs = []
                            etiquettes = []
                            for idx in counts.index:
                                if idx == 0:
                                    couleurs.append("#ef4444")  # Rouge moderne pour faux
                                    etiquettes.append("Faux billets")
                                else:
                                    couleurs.append("#10b981")  # Vert moderne pour vrais
                                    etiquettes.append("Vrais billets")
                            
                            wedges, texts, autotexts = ax.pie(
                                counts.values,
                                labels=etiquettes,
                                autopct='%1.1f%%',
                                colors=couleurs,
                                textprops={'color': '#e8eaed', 'fontsize': 12, 'weight': 'bold'},
                                startangle=90,
                                explode=(0.05, 0.05)
                            )
                            ax.set_ylabel("")
                            ax.set_title("Distribution des Prédictions", color='#e8eaed', fontsize=16, weight='bold', pad=20)
                        else:
                            # Graphique en barres avec nouvelles couleurs
                            couleurs = []
                            etiquettes = []
                            valeurs = []
                            
                            for idx in counts.index:
                                if idx == 0:
                                    couleurs.append("#ef4444")  # Rouge moderne
                                    etiquettes.append("Faux billets")
                                else:
                                    couleurs.append("#10b981")  # Vert moderne
                                    etiquettes.append("Vrais billets")
                                valeurs.append(counts[idx])
                            
                            bars = ax.bar(etiquettes, valeurs, color=couleurs, alpha=0.9, edgecolor='#e8eaed', linewidth=1)
                            
                            # Ajouter les valeurs sur les barres avec style moderne
                            for bar, valeur in zip(bars, valeurs):
                                height = bar.get_height()
                                ax.text(bar.get_x() + bar.get_width()/2., height + max(valeurs)*0.02,
                                       f'{valeur}',
                                       ha='center', va='bottom', color='#e8eaed', fontsize=12, weight='bold')
                            
                            ax.set_title("Distribution des Prédictions", color='#e8eaed', fontsize=16, weight='bold')
                            ax.set_ylabel("Nombre", color='#e8eaed', fontsize=12)
                            ax.tick_params(colors='#e8eaed', labelsize=11)
                            ax.grid(True, alpha=0.2, color='#e8eaed', linestyle='-', linewidth=0.5)
                            ax.set_axisbelow(True)

                        ax.set_facecolor('#1f2937')
                        plt.tight_layout()
                        st.pyplot(fig)
                        plt.close()
                        st.markdown("</div>", unsafe_allow_html=True)
                        
                        # Afficher le tableau complet avec les prédictions
                        st.markdown("###  Tableau des résultats")
                        st.dataframe(df, use_container_width=True)
                        
                        # Bouton de téléchargement
                        csv_data = df.to_csv(index=False)
                        st.download_button(
                            label="Télécharger les résultats (CSV)",
                            data=csv_data,
                            file_name="predictions_billets.csv",
                            mime="text/csv"
                        )
                        
                        st.success("✅ Prédictions terminées avec succès!")
                        
                else:
                    st.error(f"❌ Erreur lors de l'appel à FastAPI: {response.status_code} - {response.text}")
                    
            except requests.exceptions.ConnectionError:
                st.error("❌ Impossible de se connecter à l'API FastAPI. Vérifiez qu'elle est démarrée sur http://127.0.0.1:8000")
            except Exception as e:
                st.error(f"❌ Erreur inattendue: {str(e)}")
    else:
        st.info("Veuillez importer un CSV pour commencer.")