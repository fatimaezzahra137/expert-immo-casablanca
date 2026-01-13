import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.metrics import mean_absolute_error, r2_score
import joblib
import os
import numpy as np
import json
import matplotlib.pyplot as plt
from datetime import datetime # <--- AJOUTEZ CETTE LIGNE

# 1. PRÉPARATION DE L'ENVIRONNEMENT
if not os.path.exists('models'):
    os.makedirs('models')
if not os.path.exists('data'):
    os.makedirs('data')

CSV_PATH = 'data/houses_data.csv'

if os.path.exists(CSV_PATH):
    # 2. CHARGEMENT ET NETTOYAGE RIGOUREUX
    df = pd.read_csv(CSV_PATH)
    print(f"📦 Données brutes : {len(df)} lignes.")
    
    df = df.drop_duplicates()
    df = df.dropna(subset=['prix_vente', 'taille_terrain', 'lat', 'lon'])
    
    # Filtres de cohérence Casablanca
    df = df[df['prix_vente'].between(200000, 30000000)]
    df = df[df['taille_terrain'].between(20, 1000)]
    df = df[df['lat'].between(33.4, 33.7) & df['lon'].between(-7.8, -7.4)]
    
    print(f"✅ Données nettoyées : {len(df)} lignes prêtes.")

    # 3. VARIABLES X ET y
    features = ['taille_terrain', 'nb_chambres', 'qualite_materiaux', 'lat', 'lon', 'etage', 'garage']
    X = df[features]
    y = df['prix_vente']

    # 4. DIVISION POUR LE TEST FINAL
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 5. CONFIGURATION DU MODÈLE XGBOOST
    model = xgb.XGBRegressor(
        n_estimators=1000,
        learning_rate=0.05,
        max_depth=8,
        subsample=0.8,
        colsample_bytree=0.8,
        objective='reg:squarederror',
        random_state=42
    )

    # 6. ANALYSE DE FIABILITÉ (CROSS-VALIDATION)
    print("⏳ Calcul de la fiabilité réelle (5-Fold Cross-Validation)...")
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(model, X, y, cv=kf, scoring='r2')
    
    fiabilite_moyenne = round(np.mean(cv_scores) * 100, 2)
    stabilite_ia = round((1 - np.std(cv_scores)) * 100, 2)

    # 7. ENTRAÎNEMENT ET ÉVALUATION
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    erreur = mean_absolute_error(y_test, predictions)

    print("-" * 35)
    print(f"🎯 Précision Globale : {fiabilite_moyenne}%")
    print(f"🛡️ Stabilité Modèle  : {stabilite_ia}%")
    print(f"📊 Erreur Moyenne    : {erreur:,.0f} MAD")
    print("-" * 35)

    # 8. SAUVEGARDE DES FICHIERS POUR APP.PY
    # Sauvegarde du modèle
    joblib.dump(model, 'models/xgb_house_model.pkl')

    # Sauvegarde des métriques en JSON
    metrics = {
        "precision_globale": fiabilite_moyenne,
        "stabilite_ia": stabilite_ia,
        "derniere_maj": datetime.now().strftime("%d/%m/%Y %H:%M")
    }
    with open('models/metrics.json', 'w') as f:
        json.dump(metrics, f)

    # Sauvegarde du graphique de performance
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, predictions, alpha=0.5, color='#1E88E5', label='Prédictions')
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2, label='Idéal')
    plt.title(f'Performance du Modèle (Précision: {fiabilite_moyenne}%)')
    plt.xlabel('Prix Réels (MAD)')
    plt.ylabel('Prix Prédits (MAD)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('models/performance_plot.png')
    
    print("💾 Tous les fichiers ont été mis à jour dans /models/")

else:
    print(f"❌ Erreur : Le fichier {CSV_PATH} n'existe pas.")