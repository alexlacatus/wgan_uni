import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
import os

# ⚠️ IMPORT NECESAR: Schimbați 'script' cu numele fișierului dumneavoastră de antrenament, dacă este diferit.
# Trebuie să importați clasa Generator și toate constantele de dimensiune.
try:
    from script import Generator, CONDITION_DIM, PRODUCT_ID_DIM, NOISE_DIM, DEVICE, QUANTITY_MAX
except ImportError:
    print("EROARE: Nu se poate importa clasa Generator sau constantele din 'script.py'.")
    print("Asigurați-vă că 'script.py' există și conține toate constantele necesare.")
    exit()

# --- CONFIGURARE GENERARE ---
NUM_SAMPLES_TO_GENERATE = 10000  # Recomandat: Un număr mare pentru sarcini analitice
BATCH_SIZE = 1024  # Generare în loturi pentru a economisi memorie
MODEL_PATH = 'generator_constrained_stable.pth'
OUTPUT_FILE = 'synthetic_ecommerce_data_final.csv'
GUMBEL_TEMP_HARD = 0.01  # O temperatură foarte mică forțează selecția 'hard' (one-hot)
# Adăugați HIDDEN_DIM, deoarece Generatorul depinde de ea
HIDDEN_DIM = 256

# --- FUNCȚIA PRINCIPALĂ DE GENERARE ---
def load_generator_and_generate():
    """Încarcă Generatorul antrenat și produce setul de date sintetic."""

    if not os.path.exists(MODEL_PATH):
        print(f"Eroare: Fișierul modelului nu a fost găsit la {MODEL_PATH}.")
        print("Asigurați-vă că antrenamentul a fost finalizat și modelul a fost salvat.")
        return

    print(f"Încărcarea Generatorului din: {MODEL_PATH}")
    G = Generator().to(DEVICE)
    G.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    G.eval()  # Setează modelul în modul de evaluare

    print(f"Model încărcat. Se generează {NUM_SAMPLES_TO_GENERATE} tranzacții sintetice...")

    synthetic_data_list = []

    with torch.no_grad():
        for i in range(0, NUM_SAMPLES_TO_GENERATE, BATCH_SIZE):
            current_batch_size = min(BATCH_SIZE, NUM_SAMPLES_TO_GENERATE - i)

            # 1. Intrări Zgomot și Condiție (C)
            noise = torch.randn(current_batch_size, NOISE_DIM, device=DEVICE)

            # *CRITIC*: Condiția (ex: ID client, Sezon) este eșantionată aleatoriu în spațiul normalizat [-1, 1].
            condition_c = 2 * torch.rand(current_batch_size, CONDITION_DIM, device=DEVICE) - 1

            # 2. Generarea Tranzacției
            fake_transactions_raw = G(noise, condition_c)

            # 3. Conversia Vectorului de Produs (P) la One-Hot Hard
            # Se ia vectorul P (înainte de ultima coloană Q)
            prod_vector_soft = fake_transactions_raw[:, CONDITION_DIM: -1]

            # Gumbel-Softmax cu tau foarte mic și hard=True pentru a obține un vector one-hot curat
            prod_vector_hard = F.gumbel_softmax(prod_vector_soft, tau=GUMBEL_TEMP_HARD, hard=True, dim=1)

            # 4. Reconstrucția Tranzacției Finale Curate
            final_features = fake_transactions_raw[:, :CONDITION_DIM]
            final_quantity_norm = fake_transactions_raw[:, -1].unsqueeze(1)

            clean_transaction = torch.cat((final_features, prod_vector_hard, final_quantity_norm), dim=1)

            synthetic_data_list.append(clean_transaction.cpu().numpy())

    # --- 5. Conversie, Denormalizare și Salvare ---
    synthetic_np = np.vstack(synthetic_data_list)

    # Coloane DataFrame
    columns = (
            [f'Cond_{i}' for i in range(CONDITION_DIM)] +
            [f'ProdID_{i}' for i in range(PRODUCT_ID_DIM)] +
            ['Quantity_Norm']
    )
    synthetic_df = pd.DataFrame(synthetic_np, columns=columns)

    # Denormalizarea Cantității: Convertirea din [0, 2] la scala reală și rotunjire la întregi
    synthetic_df['Quantity'] = (synthetic_df['Quantity_Norm'] * QUANTITY_MAX / 2.0).round(0).astype(int)

    # Eliminarea coloanei de cantitate normalizată intermediară
    synthetic_df.drop(columns=['Quantity_Norm'], inplace=True)

    # Asigură-te că valoarea minimă a Cantității este 1 (nu 0) pentru o tranzacție validă
    synthetic_df['Quantity'] = synthetic_df['Quantity'].clip(lower=1)

    print("\n--- Generare Finalizată ---")
    print(f"Total mostre generate: {len(synthetic_df)}")
    print(f"Dataframe salvat la: {OUTPUT_FILE}")
    print(f"Exemplu (primele rânduri):\n{synthetic_df.head()}")

    # Salvare în format CSV (ideal pentru analiză ulterioară)
    synthetic_df.to_csv(OUTPUT_FILE, index=False)


if __name__ == '__main__':
    load_generator_and_generate()