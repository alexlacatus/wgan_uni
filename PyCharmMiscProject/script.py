import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import grad

import pandas as pd

# --- 1. CONFIGURATION ---
# Dimensions
NOISE_DIM = 128
CONDITION_DIM = 10
PRODUCT_ID_DIM = 50
OUTPUT_DIM = CONDITION_DIM + PRODUCT_ID_DIM + 1
HIDDEN_DIM = 256
N_CRITIC = 5

# WGAN-GP Parameters
GRADIENT_PENALTY_LAMBDA = 5

# Custom Loss Parameters
QUANTITY_MAX = 50.0
STOCK_PENALTY_ALPHA = 0.0  # Disabled

MIN_QUANTITY_ALPHA = 0.0  # Disabled
VARIANCE_ALPHA = 0.0  # Disabled

# --- CRITICAL HYBRID LOSS PARAMETERS ---
FEATURE_MATCH_ALPHA = 1.0
QUANTITY_MSE_ALPHA = 5.0 # 🔥 CRITICAL: Strong direct penalty

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = 'generator_constrained_stable.pth'
REAL_DATA_PATH = 'dummy_real_data_for_eval.csv'

# --- 2. THE STOCK LOOKUP TENSOR (Simplified Setup) ---
REAL_STOCK_LEVELS = torch.randint(low=1, high=int(QUANTITY_MAX * 0.8), size=(PRODUCT_ID_DIM,), dtype=torch.float32).to(
    DEVICE)
REAL_STOCK_LEVELS_NORMALIZED = REAL_STOCK_LEVELS / QUANTITY_MAX


# --- 3. THE GENERATOR NETWORK (G) ---
class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        input_size = NOISE_DIM + CONDITION_DIM

        self.common_trunk = nn.Sequential(
            nn.Linear(input_size, HIDDEN_DIM),
            nn.ReLU(),
            nn.Linear(HIDDEN_DIM, HIDDEN_DIM),
            nn.ReLU(),
        )
        self.continuous_head = nn.Sequential(
            nn.Linear(HIDDEN_DIM, HIDDEN_DIM),
            nn.ReLU(),
            nn.Linear(HIDDEN_DIM, CONDITION_DIM),
            nn.Tanh()  # Binds output to [-1, 1]
        )

        # 2. Head for CATEGORICAL features (PRODUCT_ID_DIM)
        self.categorical_head = nn.Sequential(
            nn.Linear(HIDDEN_DIM, HIDDEN_DIM),
            nn.ReLU(),
            nn.Linear(HIDDEN_DIM, PRODUCT_ID_DIM)
            # No activation here, Gumbel-Softmax takes raw logits
        )

        # 3. Head for QUANTITY
        self.quantity_head = nn.Sequential(
            nn.Linear(HIDDEN_DIM, 1),
        )

    def forward(self, z, c):
        z_c = torch.cat((z, c), dim=1)
        trunk_output = self.common_trunk(z_c)

        # 1. Output Continuous Features
        output_continuous = self.continuous_head(trunk_output)

        # Gumbel-Softmax creates a differentiable one-hot vector
        categorical_logits = self.categorical_head(trunk_output)
        # 'hard=True' means the forward pass gets a perfect one-hot vector
        output_categorical = F.gumbel_softmax(categorical_logits, tau=1.0, hard=True)

        # 3. Output Quantity
        output_quantity_raw = self.quantity_head(trunk_output).squeeze(1)
        output_quantity = torch.clamp(torch.relu(output_quantity_raw), 0.0, 1.0)

        # 4. Combine all parts
        final_output = torch.cat((
            output_continuous,
            output_categorical,
            output_quantity.unsqueeze(1)
        ), dim=1)

        return final_output


# --- 4. THE CRITIC NETWORK (D) ---
class Critic(nn.Module):
    def __init__(self):
        super().__init__()
        input_size = OUTPUT_DIM
        self.net = nn.Sequential(
            nn.Linear(input_size, HIDDEN_DIM),
            nn.LeakyReLU(0.2),
            nn.Linear(HIDDEN_DIM, HIDDEN_DIM),
            nn.LeakyReLU(0.2),
            nn.Linear(HIDDEN_DIM, 1)
        )

    def forward(self, x):
        return self.net(x)


# --- 5. WGAN-GP UTILITY FUNCTIONS ---
def calculate_gradient_penalty(critic, real_data, fake_data, device):
    BATCH_SIZE = real_data.size(0)
    alpha = torch.rand(BATCH_SIZE, 1, device=device).expand_as(real_data)
    interpolates = (alpha * real_data + (1 - alpha) * fake_data).requires_grad_(True)
    critic_interpolates = critic(interpolates)

    gradients = grad(
        outputs=critic_interpolates,
        inputs=interpolates,
        grad_outputs=torch.ones_like(critic_interpolates),
        create_graph=True,
        retain_graph=True,
    )[0].view(BATCH_SIZE, -1)

    gradient_norm = gradients.norm(2, dim=1)
    gradient_penalty = ((gradient_norm - 1) ** 2).mean()
    return gradient_penalty


# --- 6. CUSTOM STOCK PENALTY FUNCTION ---
# This remains defined but is disabled by STOCK_PENALTY_ALPHA = 0.0

def custom_stock_penalty(fake_transactions, stock_levels_norm, alpha):
    if alpha == 0:
        return torch.tensor(0.0, device=fake_transactions.device)
    # ... (Actual calculation logic is skipped as alpha is 0)
    return torch.tensor(0.0, device=fake_transactions.device)

# --- 7. TRAINING LOOP ---
# The loss function is now a hybrid: WGAN_Adv + FeatureMatch + Quantity_MSE
def train_cgan(generator, critic, data_loader, stock_levels_norm, epochs):
    # 🔥 CRITICAL LR FIX: Balanced Generator/Critic LR
    opt_g = optim.Adam(generator.parameters(), lr=5e-6, betas=(0.5, 0.9))
    opt_d = optim.Adam(critic.parameters(), lr=5e-6, betas=(0.5, 0.9))
    generator.train()
    critic.train()

    for epoch in range(epochs):
        for batch_idx, real_transactions in enumerate(data_loader):
            real_transactions = real_transactions.to(DEVICE)
            BATCH_SIZE = real_transactions.size(0)
            real_c = real_transactions[:, :CONDITION_DIM]

            # --- Train Critic (D) ---
            for _ in range(N_CRITIC):
                opt_d.zero_grad()
                noise = torch.randn(BATCH_SIZE, NOISE_DIM, device=DEVICE)
                fake_transactions = generator(noise, real_c).detach()

                critic_real = critic(real_transactions)
                critic_fake = critic(fake_transactions)
                gp = calculate_gradient_penalty(critic, real_transactions, fake_transactions, DEVICE)

                loss_critic = -(torch.mean(critic_real) - torch.mean(critic_fake)) + GRADIENT_PENALTY_LAMBDA * gp
                loss_critic.backward()
                opt_d.step()

            # --- Train Generator (G) ---
            opt_g.zero_grad()
            noise = torch.randn(BATCH_SIZE, NOISE_DIM, device=DEVICE)
            fake_transactions = generator(noise, real_c)

            y_real_norm = real_transactions[:, -1]

            quantity_gen_norm = fake_transactions[:, -1]

            # We are comparing (0-1) vs (0-1).
            loss_quantity_mse = QUANTITY_MSE_ALPHA * torch.mean((quantity_gen_norm - y_real_norm) ** 2)

            # 1. Adversarial Loss (WGAN core)
            loss_adv = -torch.mean(critic(fake_transactions))
            loss_penalty = custom_stock_penalty(fake_transactions, stock_levels_norm, STOCK_PENALTY_ALPHA)

            # 2. Feature Matching Loss (C and P correlation)
            real_features = real_transactions[:, :-1]
            fake_features = fake_transactions[:, :-1]
            loss_feature_match = FEATURE_MATCH_ALPHA * torch.mean(
                torch.abs(real_features.mean(dim=0) - fake_features.mean(dim=0))
            )

            # 4. Total Loss
            loss_generator = loss_adv + loss_penalty + loss_feature_match + loss_quantity_mse
            loss_generator.backward()
            opt_g.step()

        # Update print statement for better monitoring
        print(
            f"Epoch {epoch + 1}/{epochs} | D Loss: {loss_critic.item():.4f} | G Loss: {loss_generator.item():.4f} (Adv: {loss_adv.item():.4f}, FM: {loss_feature_match.item():.4f}, MSE_Q: {loss_quantity_mse.item():.4f})")
# --- 8. EXECUTION ---
if __name__ == '__main__':

    # 1. Load Data
    try:
        # Load the CSV data. Assuming it has already been normalized!
        real_data_df = pd.read_csv(REAL_DATA_PATH)
        print(f"Loaded normalized data with {len(real_data_df)} rows.")
        # 🔥 CRITICAL FIX: FORCING NORMALIZATION (as requested)
        # This assumes the last column is the quantity and it's un-normalized
        quantity_col_name = real_data_df.columns[-1]

        print(f"Applying mandatory normalization to quantity column: '{quantity_col_name}'...")
        real_data_df[quantity_col_name] = real_data_df[quantity_col_name] / QUANTITY_MAX
        print("Normalization applied.")
        DUMMY_DATA = torch.tensor(real_data_df.values, dtype=torch.float32).to(DEVICE)
    except FileNotFoundError:
        print(f"ERROR: File '{REAL_DATA_PATH}' not found.")
        print("Please ensure your data is pre-processed, normalized, and saved to this path.")
        exit()

    # 2. Create DataLoader
    BATCH_SIZE = 64
    data_loader = torch.utils.data.DataLoader(DUMMY_DATA, batch_size=BATCH_SIZE, shuffle=True)

    # 3. Initialize Models and Train
    G = Generator().to(DEVICE)
    D = Critic().to(DEVICE)
    print(f"Starting FINAL re-training on {DEVICE}...")

    train_cgan(G, D, data_loader, REAL_STOCK_LEVELS_NORMALIZED, epochs=100)  # Increased epochs for final attempt

    torch.save(G.state_dict(), MODEL_PATH)
    print("-" * 50)
    print(f"Re-training finished. Generator saved to: {MODEL_PATH}")
    print("-" * 50)