import streamlit as st
import pandas as pd
import numpy as np
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
from datetime import datetime
from io import BytesIO

st.set_page_config(page_title="Valorisation Portefeuille Obligations - BAM", layout="wide")
st.title("📊 Courbe des Taux Zéro-Coupon - BAM & Valorisation Portefeuille")

# 🔹 Date de valorisation automatique (aujourd'hui)
date_eval = pd.to_datetime(datetime.today().date())
st.write(f"**Date de valorisation : {date_eval.date()}**")

# 🔗 Récupération BAM
url = "https://www.bkam.ma/Marches/Principaux-indicateurs/Marche-obligataire/Marche-des-bons-de-tresor"
try:
    tables = pd.read_html(url)
    df_bam = tables[0]
except Exception as e:
    st.error(f"Impossible de récupérer les données BAM : {e}")
    df_bam = None

if df_bam is not None:
    df_bam = df_bam[df_bam["Date d'échéance"].str.match(r'\d{2}/\d{2}/\d{4}')]
    df_bam["Date d'échéance"] = pd.to_datetime(df_bam["Date d'échéance"], dayfirst=True, errors='coerce')
    df_bam["Date de la valeur"] = pd.to_datetime(df_bam["Date de la valeur"], dayfirst=True, errors='coerce')
    df_bam = df_bam.dropna(subset=["Date d'échéance", "Date de la valeur"])
    df_bam["maturite_annees"] = (df_bam["Date d'échéance"] - df_bam["Date de la valeur"]).dt.days / 365
    df_bam["taux_actuariel"] = df_bam["Taux moyen pondéré"].str.replace("%","").str.replace(",",".").astype(float)/100

    st.subheader("Table BAM nettoyée")
    st.dataframe(df_bam[["Date d'échéance","maturite_annees","taux_actuariel"]])

    # Interpolation linéaire
    x_known = df_bam["maturite_annees"].values
    y_known = df_bam["taux_actuariel"].values
    interpolateur = interp1d(x_known, y_known, kind="linear", fill_value="extrapolate")
    maturites_cibles = np.array([0.25,0.5] + list(range(1,31)))
    taux_interp = interpolateur(maturites_cibles)
    courbe_interp = pd.DataFrame({"maturite_annees": maturites_cibles,"taux_interpole": taux_interp})

    st.subheader("Courbe des taux interpolée")
    st.line_chart(courbe_interp.set_index("maturite_annees"))

    st.subheader("Graphique détaillé")
    fig, ax = plt.subplots(figsize=(10,6))
    ax.plot(courbe_interp["maturite_annees"], courbe_interp["taux_interpole"], label="Taux interpolés", marker='o')
    ax.scatter(x_known, y_known, color='red', label="Points BAM")
    ax.set_xlabel("Maturité (années)")
    ax.set_ylabel("Taux")
    ax.set_title("Courbe des taux zéro-coupon - BAM")
    ax.grid(True)
    ax.legend()
    st.pyplot(fig)

    # 🔹 Upload du portefeuille
    st.subheader("📂 Upload fichier portefeuille obligataire (Excel)")
    uploaded_file = st.file_uploader("Choisir un fichier Excel", type=["xlsx", "xls"])

    if uploaded_file is not None:
        df_oblig = pd.read_excel(uploaded_file, skiprows=2, header=1)
        df_oblig.columns = df_oblig.columns.str.strip()

        df_oblig["Emission"] = pd.to_datetime(df_oblig["Emission"], dayfirst=True, errors='coerce')
        df_oblig["Échéance"] = pd.to_datetime(df_oblig["Échéance"], dayfirst=True, errors='coerce')

        df_oblig = df_oblig[df_oblig["Échéance"] > pd.to_datetime(date_eval)]
        df_oblig["Mr"] = (df_oblig["Échéance"] - pd.to_datetime(date_eval)).dt.days
        df_oblig["Mr_annee"] = df_oblig["Mr"]/360
        df_oblig["rN"] = interpolateur(df_oblig["Mr_annee"])

        def valoriser_oblig(row):
            N = row["Nominal"]
            rf = row["Taux Facial"]/100
            Mr = row["Mr"]
            r = row["rN"]
            Mi = (row["Échéance"] - row["Emission"]).days
            if Mi <= 360:
                C = N*rf*Mi/360
                P_dirty = (N+C)/(1+r*Mr/360)
                Cc = N*rf*(Mi-Mr)/360
                P_clean = P_dirty-Cc
            else:
                n = int(np.ceil(Mr/360))
                coupons = [N*rf]*(n-1)+[N*rf+N]
                P_dirty = sum([c/((1+r)**i) for i,c in enumerate(coupons,1)])
                Cc = N*rf*(Mi-Mr)/360
                P_clean = P_dirty-Cc
            return pd.Series([P_clean,P_dirty,Cc], index=["Prix clean","Prix dirty","Coupon couru"])

        df_oblig[["Prix clean","Prix dirty","Coupon couru"]] = df_oblig.apply(valoriser_oblig, axis=1)
        df_oblig["Valeur obligation"] = df_oblig["Prix dirty"]*df_oblig["Quantité en stock"]

        def calcul_risque(row):
            N = row["Nominal"]
            rf = row["Taux Facial"]/100
            r = row["rN"]
            Mr = row["Mr"]
            n = int(np.ceil(Mr/360))
            coupons = [N*rf]*(n-1)+[N*rf+N]
            t = np.arange(1,n+1)
            pv = np.array([c/((1+r)**ti) for c,ti in zip(coupons,t)])
            price = np.sum(pv)
            duration = np.sum(t*pv)/price
            sens = duration/(1+r)
            conv = np.sum(pv*t*(t+1))/((1+r)**2*price)
            return pd.Series([duration,sens,conv], index=["Duration","Sensibilité","Convexité"])

        df_oblig[["Duration","Sensibilité","Convexité"]] = df_oblig.apply(calcul_risque, axis=1)

        # 🔹 Totaux du portefeuille
        valeur_totale = df_oblig["Valeur obligation"].sum()
        duration_totale = np.sum(df_oblig["Duration"] * df_oblig["Valeur obligation"])/valeur_totale
        sensibilite_totale = np.sum(df_oblig["Sensibilité"] * df_oblig["Valeur obligation"])/valeur_totale
        convexite_totale = np.sum(df_oblig["Convexité"] * df_oblig["Valeur obligation"])/valeur_totale

        st.subheader("Portefeuille valorisé")
        st.dataframe(df_oblig.round(2))
        st.success(f"💰 Valeur totale du portefeuille : {valeur_totale:,.2f} MAD")
        st.info(f"📊 Duration totale : {duration_totale:.4f} | Sensibilité totale : {sensibilite_totale:.4f} | Convexité totale : {convexite_totale:.4f}")

        # 🔹 Bouton pour télécharger le portefeuille valorisé
        towrite = BytesIO()
        df_oblig.to_excel(towrite, index=False, engine='openpyxl')
        towrite.seek(0)
        st.download_button(
            label="⬇️ Télécharger portefeuille valorisé",
            data=towrite,
            file_name="portefeuille_valorise.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
