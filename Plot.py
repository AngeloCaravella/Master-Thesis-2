# ==========================================================================================
# SCRIPT DI VERIFICA PREZZI (v1.0)
#
# Descrizione:
# Questo script plotta unicamente i profili di prezzo per ogni case study,
# per verificare che i dati di input siano corretti e fedeli al paper.
# ==========================================================================================
import matplotlib.pyplot as plt
import numpy as np
import os

# Importa i profili di prezzo dallo script di addestramento
try:
    from Paper_Replication_DQN import PAPER_PRICE_PROFILE, LOW_PRICE_PROFILE
except ImportError:
    print("ERRORE: Esegui prima lo script 'Paper_Replication_DQN.py' per definire i profili.")
    exit()

def create_uncertain_profile(p, t, a):
    n={}; f=a/100.0
    for h in range(24): n[h]=p.get((h-t+24)%24,0)*(1+np.random.uniform(-f,f))
    return n

# Definizione dei case study ESATTAMENTE come nello script principale
case_studies = [
    {'id': 1, 'name': 'Price Profile for Case 1', 'prices': PAPER_PRICE_PROFILE, 'start_hour': 15},
    {'id': 2, 'name': 'Price Profile for Case 2', 'prices': LOW_PRICE_PROFILE, 'start_hour': 0},
    {'id': 3, 'name': 'Price Profile for Case 3', 'prices': PAPER_PRICE_PROFILE, 'start_hour': 15},
    {'id': 4, 'name': 'Price Profile for Case 4', 'prices': LOW_PRICE_PROFILE, 'start_hour': 0},
    {'id': 5, 'name': 'Price Profile for Case 5', 'prices': PAPER_PRICE_PROFILE, 'start_hour': 15},
    {'id': 6, 'name': 'Price Profile for Case 6', 'prices': LOW_PRICE_PROFILE, 'start_hour': 0},
    {'id': 7, 'name': 'Price Profile for Case 1 (Uncertainty)', 'prices': create_uncertain_profile(PAPER_PRICE_PROFILE, 3, 20), 'start_hour': 15},
    {'id': 8, 'name': 'Price Profile for Case 5 (Uncertainty)', 'prices': create_uncertain_profile(PAPER_PRICE_PROFILE, 3, 20), 'start_hour': 15},
]

results_dir = "price_verification_plots"
os.makedirs(results_dir, exist_ok=True)

print(f"--- Generazione grafici di verifica prezzi in '{results_dir}' ---")

for case in case_studies:
    price_profile = case['prices']
    
    hours = np.arange(0, 24, 0.25) # 96 step da 15 minuti
    prices = [price_profile.get(int(h), 0) * 100 for h in hours] # Prezzi in Cents/kWh

    plt.figure(figsize=(12, 6))
    plt.step(hours, prices, where='post', linewidth=2.5)
    plt.title(case['name'], fontsize=16)
    plt.xlabel("Time (Hours)", fontsize=12)
    plt.ylabel("Price (Cents/kWh)", fontsize=12)
    plt.grid(True, linestyle=':')
    plt.xlim(0, 24)
    
    filename = os.path.join(results_dir, f"price_case_{case['id']}.png")
    plt.savefig(filename)
    plt.close()
    print(f"Salvato: {filename}")

print("--- Verifica completata. ---")
