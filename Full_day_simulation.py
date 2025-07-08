# ==========================================================================================
# SIMULAZIONE CONTINUA DI 24 ORE (v1.2 - CON ANALISI COMPARATIVA)
#
# Descrizione:
# Aggiunta un'opzione "(5) Esegui tutte" nel menu per lanciare la simulazione
# di 24 ore per tutte e 4 le strategie in sequenza, usando gli stessi parametri.
# Questo permette un'analisi di sensibilità e un confronto diretto dei risultati.
# ==========================================================================================
import torch, numpy as np, matplotlib.pyplot as plt, os, sys
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

try:
    from Paper_Replication_DQN import (
        DQNAgent, 
        V2G_Environment_Paper_Simple, 
        V2G_Environment_Paper_Faithful, 
        V2G_Environment_Paper_100_Percent_Faithful,
        V2G_Environment_Paper_Debug_Idle,
        PAPER_PARAMS, 
        MODEL_PATH_SIMPLE, 
        MODEL_PATH_FAITHFUL,
        MODEL_PATH_100_FAITHFUL,
        MODEL_PATH_DEBUG_IDLE,
        PAPER_PRICE_PROFILE, 
        LOW_PRICE_PROFILE
    )
except ImportError:
    print("ERRORE: Impossibile trovare 'Paper_Replication_DQN.py'. Assicurati che sia nella stessa cartella.", file=sys.stderr)
    sys.exit(1)

ENV_CLASSES = {
    '1': V2G_Environment_Paper_Simple,
    '2': V2G_Environment_Paper_Faithful,
    '3': V2G_Environment_Paper_100_Percent_Faithful,
    '4': V2G_Environment_Paper_Debug_Idle
}
MODEL_PATHS = {
    '1': MODEL_PATH_SIMPLE,
    '2': MODEL_PATH_FAITHFUL,
    '3': MODEL_PATH_100_FAITHFUL,
    '4': MODEL_PATH_DEBUG_IDLE
}
STRATEGY_NAMES = {
    '1': 'Simple',
    '2': 'Faithful',
    '3': '100% Faithful',
    '4': 'Debug Idle'
}

def create_uncertain_profile(p, t, a):
    n = {}; f = a / 100.0
    for h in range(24):
        price = p.get((h - t + 24) % 24, 0) * (1 + np.random.uniform(-f, f))
        n.update({h: price})
    return n

PRICE_PROFILES = {
    '1': ('High Prices (Paper)', PAPER_PRICE_PROFILE),
    '2': ('Low Prices (Random)', LOW_PRICE_PROFILE),
    '3': ('High Prices (Paper)', PAPER_PRICE_PROFILE),
    '4': ('Low Prices (Random)', LOW_PRICE_PROFILE),
    '5': ('High Prices (Paper)', PAPER_PRICE_PROFILE),
    '6': ('Low Prices (Random)', LOW_PRICE_PROFILE),
    '7': ('High Prices with Uncertainty', create_uncertain_profile(PAPER_PRICE_PROFILE, 3, 20)),
    '8': ('High Prices with Uncertainty (Case 5)', create_uncertain_profile(PAPER_PRICE_PROFILE, 3, 20))
}

def run_full_day_simulation(agent, env, initial_soc, target_soc):
    history = {'soc': [initial_soc], 'actions': [], 'power': [], 'time_hours': []}
    state = env.reset(initial_soc, 'soc', target_soc, max_duration_minutes=24*60, start_hour=0)
    power_map = {0: 0.0, 1: PAPER_PARAMS['P_CONVENTIONAL'], 2: PAPER_PARAMS['P_FAST'], 3: -PAPER_PARAMS['P_V2G']}
    for step in range(96):
        time_now = env.current_hour + env.current_minute / 60.0
        history['time_hours'].append(time_now)
        action = agent.choose_action(state)
        history['actions'].append(action)
        if isinstance(env, V2G_Environment_Paper_100_Percent_Faithful) and env.is_in_forced_charge:
             history['power'].append(power_map[0])
        else:
             history['power'].append(power_map[action])
        next_state, _, _ = env.step(action)
        history['soc'].append(env.soc)
        state = next_state
    return history

def plot_full_day_simulation(history, price_profile, title, filename):
    sim_hours = np.array(history['time_hours'])
    sim_power = np.array(history['power'])
    sim_soc = np.array(history['soc'])
    fig, ax1 = plt.subplots(figsize=(20, 10))
    full_day_hours = np.arange(0, 24, 0.25)
    full_day_prices = [price_profile.get(int(h % 24), 0) * 100 for h in full_day_hours]
    ax1.plot(full_day_hours, full_day_prices, color='tab:blue', linewidth=2.5, label='Price Profile')
    ax1.set_xlabel('Time (Hours)', fontsize=14)
    ax1.set_ylabel('Price (Cents/kWh)', color='tab:blue', fontsize=14)
    ax1.tick_params(axis='y', labelcolor='tab:blue', labelsize=12)
    ax1.tick_params(axis='x', labelsize=12)
    ax1.grid(True, which='both', linestyle=':', linewidth=0.7)
    ax1.set_xlim(-0.5, 24.5)
    ax2 = ax1.twinx()
    ax2.set_ylabel('Charging Power (kW)', color='tab:orange', fontsize=14)
    ax3 = ax1.twinx()
    ax3.spines['right'].set_position(('outward', 60))
    ax3.set_ylabel('State of Charge (SoC %)', color='purple', fontsize=14)
    soc_percent = [s * 100 for s in history['soc']]
    ax3.plot(history['time_hours'] + [24.0], soc_percent, color='purple', linewidth=3, marker='o', markersize=4, linestyle='--', label='SoC')
    ax3.tick_params(axis='y', labelcolor='purple', labelsize=12)
    ax3.set_ylim(0, 105)
    colors = {1: 'orange', 2: 'red', 3: 'green'}
    idle_bar_color = 'dimgrey'
    bar_width = 0.95 * (PAPER_PARAMS['TIME_PERIOD_MINUTES'] / 60.0)
    idle_bar_height = 1.0 
    for i in range(len(sim_hours)):
        action = history['actions'][i]
        power = history['power'][i]
        time = sim_hours[i]
        if action == 0:
            ax2.bar(time, height=idle_bar_height, width=bar_width, bottom=-idle_bar_height / 2, color=idle_bar_color, alpha=0.8, edgecolor='black', linewidth=0.5, align='edge', zorder=10)
        else:
            ax2.bar(time, power, width=bar_width, color=colors[action], alpha=0.8, edgecolor='black', linewidth=0.5, align='edge')
    ax2.tick_params(axis='y', labelcolor='tab:orange', labelsize=12)
    ax2.axhline(0, color='black', linewidth=1.5, linestyle='-')
    ax2.set_ylim(-15, 30)
    plt.title(title, fontsize=18, fontweight='bold')
    legend_elements = [
        Line2D([0], [0], color='tab:blue', lw=2.5, label='Price'),
        Line2D([0], [0], color='purple', lw=3, linestyle='--', label='SoC'),
        Patch(facecolor='red', edgecolor='black', alpha=0.8, label='Fast Charging'),
        Patch(facecolor='orange', edgecolor='black', alpha=0.8, label='Conv. Charging'),
        Patch(facecolor='green', edgecolor='black', alpha=0.8, label='V2G'),
        Patch(facecolor=idle_bar_color, edgecolor='black', alpha=0.8, label='Idle')
    ]
    ax1.legend(handles=legend_elements, loc='upper left', fontsize=12)
    fig.tight_layout()
    plt.savefig(filename)
    plt.close(fig)
    print(f"Grafico della simulazione di 24 ore salvato in: {filename}")

def get_user_choice(prompt, options):
    while True:
        print(prompt)
        for key, value in options.items():
            print(f" ({key}) {value}")
        choice = input("Scelta: ")
        if choice in options:
            return choice
        print("\nScelta non valida. Riprova.\n")

def main():
    # =================================================================================
    # MODIFICA: Aggiunta l'opzione per l'analisi comparativa
    # =================================================================================
    strategy_options = {key: val for key, val in STRATEGY_NAMES.items()}
    strategy_options['5'] = 'Esegui tutte (Analisi Comparativa)'
    
    strategy_choice = get_user_choice("\nScegli la strategia (modello) da simulare:", strategy_options)
    
    # --- Selezione del Profilo di Prezzo e Parametri ---
    price_options = {key: val[0] for key, val in PRICE_PROFILES.items()}
    price_choice = get_user_choice("\nScegli lo scenario di prezzo per la simulazione:", price_options)
    price_name, price_profile = PRICE_PROFILES[price_choice]

    while True:
        try:
            initial_soc = float(input("\nInserisci lo Stato di Carica (SoC) iniziale (es. 0.3 per 30%): "))
            if 0.0 <= initial_soc <= 1.0: break
            print("Valore non valido. Inserisci un numero tra 0.0 e 1.0.")
        except ValueError: print("Input non valido. Inserisci un numero.")

    while True:
        try:
            target_soc = float(input("Inserisci lo Stato di Carica (SoC) target (es. 0.9 per 90%): "))
            if 0.0 <= target_soc <= 1.0: break
            print("Valore non valido. Inserisci un numero tra 0.0 e 1.0.")
        except ValueError: print("Input non valido. Inserisci un numero.")

    # --- Logica di Esecuzione ---
    if strategy_choice == '5':
        # Esegue tutte le strategie in sequenza
        print("\n--- AVVIO ANALISI COMPARATIVA DI TUTTE LE STRATEGIE ---")
        strategies_to_run = STRATEGY_NAMES.keys()
    else:
        # Esegue solo la strategia scelta
        strategies_to_run = [strategy_choice]

    for choice in strategies_to_run:
        env_class = ENV_CLASSES[choice]
        model_path = MODEL_PATHS[choice]
        strategy_name = STRATEGY_NAMES[choice]

        print(f"\n--- ESECUZIONE SIMULAZIONE PER LA STRATEGIA: {strategy_name} ---")

        if not os.path.exists(model_path):
            print(f"ATTENZIONE: Modello '{model_path}' non trovato. Salto questa strategia.", file=sys.stderr)
            continue

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        agent = DQNAgent(device)
        agent.load(model_path)
        agent.e = 0.0

        env = env_class(price_profile)
        history = run_full_day_simulation(agent, env, initial_soc, target_soc)

        results_dir = "full_day_results"
        os.makedirs(results_dir, exist_ok=True)
        
        # Crea un nome di file che include la strategia e il profilo di prezzo
        price_prefix = price_name.split(' ')[0].lower()
        filename = f"24h_sim_{strategy_name.lower().replace(' ', '_')}_price_{price_prefix}.png"
        title = f"24h Simulation | Strategy: {strategy_name} | Prices: {price_name}\nInitial SoC: {initial_soc*100:.0f}% | Target SoC: {target_soc*100:.0f}%"
        
        plot_full_day_simulation(history, price_profile, title, os.path.join(results_dir, filename))

    print("\n--- SIMULAZIONE COMPLETATA ---")

if __name__ == "__main__":
    main()
