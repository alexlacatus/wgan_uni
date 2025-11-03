import torch
import torch.nn as nn
import torch.nn.functional as F
from flask import Flask, jsonify, request
import numpy as np
import os
import random

# ⚠️ IMPORTĂ DEFINIȚIILE MODELULUI ȘI CONSTANTELE
# Asigură-te că Generator și constantele sunt importate din scriptul tău principal
try:
    from script import Generator, NOISE_DIM, CONDITION_DIM, PRODUCT_ID_DIM, DEVICE, QUANTITY_MAX
except ImportError:
    print("Eroare la import: Asigură-te că Generatorul și constantele sunt definite în 'script.py'.")
    exit()

# --- CONFIGURARE API & MODEL ---
app = Flask(__name__)
MODEL_PATH = 'generator_constrained_stable.pth'  # Calea către modelul antrenat
GUMBEL_TEMP_HARD = 0.01

# Variabila globală pentru model
G = None


def load_gan_model():
    """Încarcă modelul Generator la pornirea aplicației."""
    global G
    print("Încărcare model GAN...")

    # Asigură-te că toate constantele sunt disponibile (dacă nu au fost importate de la script)
    # Dacă nu le poți importa, le definești aici, exemplu: CONDITION_DIM = 10

    G = Generator().to(DEVICE)
    try:
        G.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
        G.eval()
        print("Model GAN încărcat cu succes.")
    except FileNotFoundError:
        print(f"Eroare: Modelul nu a fost găsit la {MODEL_PATH}")
        G = None


# --- RUTA API: GENERARE DATE ---
@app.route('/generate', methods=['POST'])
def generate_data():
    if G is None:
        return jsonify({"error": "Modelul nu este încărcat."}), 503

    try:
        # Preluarea numărului de mostre din cererea POST
        data = request.get_json()
        num_samples = int(data.get('num_samples', 1))

        if num_samples <= 0 or num_samples > 100:
            return jsonify({"error": "Numărul de mostre trebuie să fie între 1 și 100."}), 400

        synthetic_data_list = []

        with torch.no_grad():
            for _ in range(num_samples):
                # 1. Intrări: Zgomot și Condiție
                noise = torch.randn(1, NOISE_DIM, device=DEVICE)
                # Condiția C este eșantionată aleatoriu (ca și în scriptul de generare)
                condition_c = 2 * torch.rand(1, CONDITION_DIM, device=DEVICE) - 1

                # 2. Generarea Tranzacției
                fake_transactions_raw = G(noise, condition_c)

                # 3. Conversia Produsului la One-Hot Hard (Gumbel-Softmax)
                prod_vector_soft = fake_transactions_raw[:, CONDITION_DIM: -1]
                prod_vector_hard = F.gumbel_softmax(prod_vector_soft, tau=GUMBEL_TEMP_HARD, hard=True, dim=1)

                # 4. Decodificare (pentru a fi mai ușor de citit în Spring Boot)

                # 4a. Cantitatea Denormalizată (din [0, 2] la scala reală)
                final_quantity_norm = fake_transactions_raw[:, -1].item()
                quantity = int(np.clip(np.round(final_quantity_norm * QUANTITY_MAX / 2.0), 1, QUANTITY_MAX))

                # 4b. ProductID Decodat
                product_id_index = torch.argmax(prod_vector_hard).item()

                # 4c. Condițiile
                conditions = final_features = fake_transactions_raw[:, :CONDITION_DIM].squeeze().tolist()

                # Colectare rezultat JSON
                synthetic_data_list.append({
                    "quantity": quantity,
                    "productId": product_id_index,
                    "conditions": conditions  # Rămân în format normalizat [-1, 1]
                })

        return jsonify(synthetic_data_list)

    except Exception as e:
        print(f"Eroare la generare: {e}")
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    load_gan_model()  # Încărcați modelul înainte de a porni serverul
    app.run(host='0.0.0.0', port=5000)