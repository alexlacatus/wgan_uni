import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# --- 1. CONFIGURAȚIE ȘI CĂI DE FIȘIERE ---
SYNTHETIC_DATA_PATH = 'synthetic_ecommerce_data_final.csv'
# ⚠️ ADAPTAȚI ACESTEA: Calea către fișierul dumneavoastră de date reale (preprocesate!)
REAL_DATA_PATH = 'dummy_real_data_for_eval.csv'
OUTPUT_EVAL_FILE = 'evaluation_results.txt'

# Constante de dimensiune (folosite pentru a identifica coloanele)
CONDITION_DIM = 10
PRODUCT_ID_DIM = 50


# --- 2. FUNCȚII DE PREGĂTIRE A DATELOR ---

def prepare_data(df, is_synthetic=False):
    """Pregătește DataFrame-ul pentru antrenare (X și y) folosind coloanele One-Hot."""

    # NU mai facem nicio decodificare aici!

    # Definiți X (Intrări) și y (Ieșire)
    cond_cols = [c for c in df.columns if c.startswith('Cond_')]
    prod_cols = [c for c in df.columns if c.startswith('ProdID_')]

    # X va conține Coloanele Condiționale ȘI Coloanele One-Hot ale Produsului
    X = df[cond_cols + prod_cols]
    y = df['Quantity']

    # Asigurarea tipului de date corect
    if X.select_dtypes(include='object').empty:
        X = X.astype(np.float32)

    return X, y


# --- 3. FUNCȚIE DE EVALUARE A PERFORMANȚEI ---

def evaluate_model(model, X_train, y_train, X_test, y_test, name):
    """Antrenează și evaluează modelul de prognoză a cererii."""

    print(f"\n--- Antrenare Model ({name}) ---")

    # Random Forest este un model robust și standard pentru comparație
    model.fit(X_train, y_train)

    # Predicție pe setul de test comun
    predictions = model.predict(X_test)

    # Asigură-te că predicțiile sunt pozitive (o cantitate nu poate fi negativă)
    predictions = np.maximum(0, predictions)

    # Metricile cheie: RMSE și R-squared
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    r2 = r2_score(y_test, predictions)

    results = {
        'Dataset': name,
        'RMSE': rmse,
        'R2_Score': r2
    }

    return results


# --- 4. RULAREA PRINCIPALĂ A EVALUĂRII ---

if __name__ == '__main__':

    # 1. Încărcarea Seturilor de Date
    try:
        synthetic_df = pd.read_csv(SYNTHETIC_DATA_PATH)
        real_df = pd.read_csv(REAL_DATA_PATH)
    except FileNotFoundError as e:
        print(f"EROARE LA ÎNCĂRCARE: {e}. Vă rugăm să verificați căile de fișiere.")
        exit()

    # 2. Pregătirea Datelor
    X_synth, y_synth = prepare_data(synthetic_df, is_synthetic=True)
    X_real, y_real = prepare_data(real_df, is_synthetic=False)

    # 3. Crearea unui Set de Test Comun (pentru comparație corectă)
    # Folosim setul real ca sursă de adevăr pentru test
    X_test_real, _, y_test_real, _ = train_test_split(
        X_real, y_real, test_size=0.8, random_state=42
    )

    # 4. Antrenare și Evaluare

    # A. Model antrenat pe DATE REALE
    model_real = RandomForestRegressor(n_estimators=100, random_state=42)
    results_real = evaluate_model(model_real, X_real, y_real, X_test_real, y_test_real, "REAL DATA")

    # B. Model antrenat pe DATE SINTETICE
    model_synth = RandomForestRegressor(n_estimators=100, random_state=42)
    results_synth = evaluate_model(model_synth, X_synth, y_synth, X_test_real, y_test_real, "SYNTHETIC DATA")

    # 5. Afișarea și Salvarea Rezultatelor

    final_df = pd.DataFrame([results_real, results_synth])

    # Comparare finală pentru raport
    comparison = f"""
    =======================================================
    | RAPORT FINAL DE EVALUARE A CALITĂȚII DATELOR GAN |
    =======================================================

    Obiectiv: A demonstra că setul sintetic (antrenat) obține o performanță similară
              cu setul real (antrenat) atunci când testează pe date reale.

    Metodă: RandomForestRegressor (100 estimatori)
    Metrică Cheie: RMSE (Root Mean Squared Error)

    -------------------------------------------------------
    | Indicatori de Succes:
    | RMSE_SINTETIC ar trebui să fie apropiat de RMSE_REAL.
    -------------------------------------------------------

    {final_df.to_string(index=False)}

    =======================================================
    """

    print(comparison)

    with open(OUTPUT_EVAL_FILE, 'w', encoding='utf-8') as f:
        f.write(comparison)

    print(f"\nRezultatele complete au fost salvate în: {OUTPUT_EVAL_FILE}")