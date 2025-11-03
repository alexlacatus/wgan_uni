import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder

from script import CONDITION_DIM, PRODUCT_ID_DIM


def generate_dummy_real_data(num_rows, cond_dim, prod_dim, quantity_max):
    # Generare Condiții (Normalizate similar Generatorului)
    cond_data = np.random.uniform(-1, 1, size=(num_rows, cond_dim))
    cond_df = pd.DataFrame(cond_data, columns=[f'Cond_{i}' for i in range(cond_dim)])

    # Generare Produse (Codificare One-Hot)
    prod_indices = np.random.randint(0, prod_dim, num_rows)
    prod_oh_encoder = OneHotEncoder(sparse_output=False, categories=[range(prod_dim)])
    prod_oh = prod_oh_encoder.fit_transform(prod_indices.reshape(-1, 1))
    prod_df = pd.DataFrame(prod_oh, columns=[f'ProdID_{i}' for i in range(prod_dim)])

    # Ipotetic (Definirea unei relații de bază plauzibile)
    # Cantitatea depinde de primele două condiții și de un factor aleatoriu
    base_quantity = 25 + 20 * (cond_df['Cond_0'] + 0.5 * cond_df['Cond_1'])
    noise = np.random.randint(-2, 3, size=num_rows)
    quantity = np.clip(np.round(base_quantity + noise), 1, quantity_max)

    # Crearea DataFrame-ului final
    dummy_df = pd.concat([cond_df, prod_df], axis=1)
    dummy_df['Quantity'] = quantity.astype(int)

    return dummy_df


# Generați setul de date artificial
if __name__ == '__main__':
    dummy_real_df = generate_dummy_real_data(
        num_rows=10000,
        cond_dim=CONDITION_DIM,
        prod_dim=PRODUCT_ID_DIM,
        quantity_max=50
    )
    dummy_real_df.to_csv('dummy_real_data_for_eval.csv', index=False)