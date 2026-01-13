import streamlit as st
import pandas as pd
import joblib
import folium
import numpy as np
import unicodedata
import json
import os
import plotly.express as px
from streamlit_folium import st_folium
from fpdf import FPDF
from fpdf.enums import XPos, YPos
from geopy.geocoders import Nominatim
from datetime import datetime



# --- CONFIGURATION PAGE ---

st.set_page_config(page_title="Expert Immo Casablanca", layout="wide")

if 'prix_calcule' not in st.session_state:
    st.session_state.prix_calcule = None
if 'lat' not in st.session_state:
    st.session_state.lat, st.session_state.lon = 33.5731, -7.5898



# --- FONCTIONS TECHNIQUES ---

def clean_for_pdf(text):
    """Supprime les accents pour la génération du PDF"""
    if not text: return "Casablanca"
    nfkd_form = unicodedata.normalize('NFKD', str(text))
    return "".join([c for c in nfkd_form if not unicodedata.combining(c)]).encode('ascii', 'ignore').decode('ascii')



def generate_pdf_data(prix, surf, ch, qual, zone_name):

    """Génère le rapport PDF avec note de fiabilité"""

    pdf = FPDF()
    pdf.add_page()

   

    # Header

    pdf.set_font("helvetica", 'B', 16)
    pdf.cell(200, 10, "RAPPORT D'ESTIMATION IMMOBILIERE", align='C', new_x=XPos.LMARGIN, new_y=YPos.NEXT)

   

    # Infos Bien

    pdf.ln(10)
    pdf.set_font("helvetica", 'B', 12)

    pdf.set_text_color(40, 70, 150)
    pdf.cell(0, 10, f"LOCALISATION : {clean_for_pdf(zone_name)}", new_x=XPos.LMARGIN, new_y=YPos.NEXT)


    pdf.set_text_color(0, 0, 0)
    pdf.set_font("helvetica", '', 11)
    pdf.cell(0, 8, f"Surface : {surf} m2 | Chambres : {ch} | Qualite : {qual}/10", new_x=XPos.LMARGIN, new_y=YPos.NEXT)

   

    # Résultat

    pdf.ln(10)
    pdf.set_fill_color(230, 230, 230)
    pdf.set_font("helvetica", 'B', 15)
    pdf.cell(0, 15, f"VALEUR ESTIMEE : {prix:,.0f} MAD", border=1, fill=True, align='C', new_x=XPos.LMARGIN, new_y=YPos.NEXT)

   

    # Note de Fiabilité
    pdf.ln(15)
    pdf.set_font("helvetica", 'B', 10)
    pdf.set_text_color(100, 100, 100)
    pdf.cell(0, 8, "NOTE DE FIABILITE ET METHODOLOGIE", new_x=XPos.LMARGIN, new_y=YPos.NEXT)

   

    pdf.set_font("helvetica", 'I', 9)
    note = ("Cette expertise est generee par un modele d'intelligence artificielle specialise sur le marche "
            "de Casablanca. Elle utilise la methode du Gradient Boosting pour croiser les caracteristiques "
            "techniques du bien avec les dynamiques de prix locales, garantissant une estimation objective "
            "basee sur la donnee reelle du marche.")
    pdf.multi_cell(0, 5, clean_for_pdf(note))



    # Contact

    pdf.ln(10)
    pdf.set_font("helvetica", 'B', 10)
    pdf.set_text_color(200, 0, 0)
    pdf.cell(0, 8, "SUPPORT & EXPERTISE : contact.expert@casa-immo.ma", new_x=XPos.LMARGIN, new_y=YPos.NEXT)

   

    return bytes(pdf.output())



def save_feedback(quartier, prix_ia, avis, surface, prix_espere=None):
    """Sauvegarde locale des avis dans le fichier CSV"""
    feedback_file = 'data/feedbacks.csv'
    new_data = pd.DataFrame([{
        'date': datetime.now().strftime("%Y-%m-%d %H:%M"),
        'quartier': quartier, 'prix_estime': int(prix_ia),
        'surface': surface, 'avis': avis, 'prix_espere': prix_espere

    }])
    if os.path.exists(feedback_file):
        new_data.to_csv(feedback_file, mode='a', header=False, index=False)
    else:
        new_data.to_csv(feedback_file, index=False)



# --- CHARGEMENT ---

@st.cache_resource
def load_model():
    return joblib.load('models/xgb_house_model.pkl')



model = load_model()



# --- INTERFACE ---

st.title("📍 ImmoPredict Casablanca")
tab1, tab2 = st.tabs(["🔍 Estimateur", "🔐 Admin & Stats"])



with tab1:
    col_map, col_form = st.columns([2, 1])
    with col_map:
        m = folium.Map(location=[st.session_state.lat, st.session_state.lon], zoom_start=13)
        folium.Marker([st.session_state.lat, st.session_state.lon], icon=folium.Icon(color='red')).add_to(m)
        map_data = st_folium(m, width=700, height=500, key="map_casa")
        if map_data and map_data['last_clicked']:

            st.session_state.lat, st.session_state.lon = map_data['last_clicked']['lat'], map_data['last_clicked']['lng']



    with col_form:
        st.subheader("⚙️ Paramètres")
        surf = st.number_input("Surface (m2)", 20, 1000, 100)
        ch = st.slider("Chambres", 1, 10, 3)
        qual = st.select_slider("Qualité", range(1, 11), 5)

       

        # Identification du quartier

        geolocator = Nominatim(user_agent="expert_casa_vfinal")
        try:
            loc = geolocator.reverse(f"{st.session_state.lat}, {st.session_state.lon}")
            nom_q = loc.raw['address'].get('suburb') or loc.raw['address'].get('neighbourhood') or "Casablanca"
        except: nom_q = "Casablanca"

       

        st.success(f"📍 Quartier : **{nom_q}**")



        if st.button("💰 Calculer le prix",width='stretch'):

            input_df = pd.DataFrame([[surf, ch, qual, st.session_state.lat, st.session_state.lon, 1, 1]],

                                     columns=['taille_terrain', 'nb_chambres', 'qualite_materiaux', 'lat', 'lon', 'etage', 'garage'])

            st.session_state.prix_calcule = model.predict(input_df)[0]



        if st.session_state.prix_calcule:
            st.metric("Prix Estimé", f"{st.session_state.prix_calcule:,.0f} MAD")

           

            # PDF

            pdf_bytes = generate_pdf_data(st.session_state.prix_calcule, surf, ch, qual, nom_q)
            st.download_button("📥 Rapport PDF", pdf_bytes, f"Expertise_{nom_q}.pdf", "application/pdf",width='stretch')

           

            # Retours Utilisateurs (Sans Email)
            st.write("📢 **Votre avis :**")
            c1, c2 = st.columns(2)
            with c1:
                if st.button("👍 Correct", width='stretch'):
                    save_feedback(nom_q, st.session_state.prix_calcule, "Correct", surf)
                    st.toast("Merci pour votre retour !")
            with c2:
                with st.popover("👎 Erreur", width='stretch'):
                    p_e = st.number_input("Prix espéré ?", min_value=100000)
                    if st.button("Valider l'avis"):
                        save_feedback(nom_q, st.session_state.prix_calcule, "Incorrect", surf, p_e)
                        st.success("Avis enregistré dans le système.")



# --- ONGLET ADMIN ---

with tab2:

    st.header("🔐 Espace Administrateur")

    pwd = st.text_input("Mot de passe :", type="password")


    if pwd == "FATI1234":
        if os.path.exists('data/feedbacks.csv'):
            df_f = pd.read_csv('data/feedbacks.csv')

           
            # Graphique
            st.subheader("📊 Analyse de Satisfaction")
            fig = px.pie(df_f, names='avis', hole=0.4,
                         color='avis', color_discrete_map={'Correct':'#2ecc71', 'Incorrect':'#e74c3c'})
            st.plotly_chart(fig, width='stretch')

           
            # Tableau
            st.subheader("📋 Liste des Feedbacks")
            st.dataframe(df_f, width='stretch')

            # --- SUPPRESSION -
            st.markdown("---")
            st.subheader("🗑️ Gestion des données")
            options_suppression = df_f.index.tolist()

           

            col_del1, col_del2 = st.columns([1, 2])
            with col_del1:
                idx_to_delete = st.selectbox("Sélectionner l'index :", options_suppression)
            with col_del2:
                st.write("")
                st.write("")
                if st.button("Confirmer la suppression", type="primary", width='stretch'):
                    df_updated = df_f.drop(idx_to_delete)
                    df_updated.to_csv('data/feedbacks.csv', index=False)
                    st.success(f"Ligne {idx_to_delete} supprimée.")
                    st.rerun()
        else:
            st.info("Aucune donnée enregistrée.")