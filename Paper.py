# ==========================================================================================
# IMPLEMENTAZIONE DEI CASE STUDY DAL PAPER (v2.6 - CON OPZIONE DEBUG IDLE)
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
        V2G_Environment_Paper_Debug_Idle, # MODIFICA: Importa la nuova classe
        PAPER_PARAMS, 
        MODEL_PATH_SIMPLE, 
        MODEL_PATH_FAITHFUL,
        MODEL_PATH_100_FAITHFUL,
        MODEL_PATH_DEBUG_IDLE, # MODIFICA: Importa il nuovo path
        PAPER_PRICE_PROFILE, 
        LOW_PRICE_PROFILE
    )
except ImportError:
    print("ERRORE: Impossibile trovare 'Paper_Replication_DQN.py'.", file=sys.stderr); sys.exit(1)

# ... (FUNZIONI run_charging_session e plot_and_save_case_study INVARIATE) ...
def run_charging_session(agent, env):
    history = {'soc': [env.soc], 'actions': [], 'power': [], 'time_hours': []}
    state = env.reset(env.soc, env.mode, env.target_soc, env.max_duration_minutes, start_hour=env.initial_hour)
    done = False
    power_map = {0: 0.0, 1: PAPER_PARAMS['P_CONVENTIONAL'], 2: PAPER_PARAMS['P_FAST'], 3: -PAPER_PARAMS['P_V2G']}
    while not done:
        history['time_hours'].append(env.current_hour + env.current_minute / 60.0)
        action = agent.choose_action(state)
        history['actions'].append(action)
        if isinstance(env, V2G_Environment_Paper_100_Percent_Faithful) and env.is_in_forced_charge:
             history['power'].append(power_map[0])
        else:
             history['power'].append(power_map[action])
        next_state, _, done = env.step(action)
        history['soc'].append(env.soc)
        state = next_state
        if len(history['time_hours']) > 96 * 2:
            print("Attenzione: Simulazione interrotta per eccessiva durata.")
            break
    return history
def plot_and_save_case_study(history, price_profile, title, filename):
    if not history['time_hours']:
        print(f"Attenzione: Dati insufficienti per plottare '{title}'.")
        return
    sim_hours = np.array(history['time_hours'])
    sim_power = np.array(history['power'])
    sim_soc = np.array(history['soc'])
    fig, ax1 = plt.subplots(figsize=(20, 10))
    full_day_hours = np.arange(0, 24.25, 0.25)
    full_day_prices = [price_profile.get(int(h % 24), 0) * 100 for h in full_day_hours]
    ax1.plot(full_day_hours, full_day_prices, color='gray', linestyle='--', alpha=0.5, label='Full Day Price Profile (Context)')
    active_prices = [price_profile.get(int(h % 24), 0) * 100 for h in sim_hours]
    ax1.step(sim_hours, active_prices, where='post', color='tab:blue', linewidth=2.5, label='Active Price During Simulation')
    ax1.set_xlabel('Time (Hours)', fontsize=14)
    ax1.set_ylabel('Price (Cents/kWh)', color='tab:blue', fontsize=14)
    ax1.tick_params(axis='y', labelcolor='tab:blue', labelsize=12)
    ax1.tick_params(axis='x', labelsize=12)
    ax1.grid(True, which='both', linestyle=':', linewidth=0.7)
    ax2 = ax1.twinx()
    ax2.set_ylabel('Charging Power (kW)', color='tab:orange', fontsize=14)
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
    initial_soc_label = f"Start: {sim_soc[0]*100:.0f}%"
    ax2.text(sim_hours[0] - bar_width, 1, initial_soc_label, ha='right', va='bottom', fontsize=10, fontweight='bold', color='black')
    for i in range(len(sim_hours)):
        offset = 1.5 if sim_power[i] >= 0 else -3.0
        soc_label = f"{sim_soc[i+1]*100:.0f}%"
        ax2.text(sim_hours[i] + bar_width/2, sim_power[i] + offset, soc_label, ha='center', va='bottom', fontsize=10, fontweight='bold')
    ax1.set_xlim(sim_hours[0] - 1.0, sim_hours[-1] + 1.0)
    plt.title(title, fontsize=18, fontweight='bold')
    legend_elements = [
        Line2D([0], [0], color='tab:blue', lw=2.5, label='Active Price'),
        Line2D([0], [0], color='gray', linestyle='--', alpha=0.5, lw=2, label='Price Profile (Context)'),
        Line2D([0], [0], color='black', lw=1.5, label='Zero Power Level'),
        Patch(facecolor='red', edgecolor='black', alpha=0.8, label=f'Fast Charging ({PAPER_PARAMS["P_FAST"]} kW)'),
        Patch(facecolor='orange', edgecolor='black', alpha=0.8, label=f'Conv. Charging ({PAPER_PARAMS["P_CONVENTIONAL"]} kW)'),
        Patch(facecolor='green', edgecolor='black', alpha=0.8, label=f'V2G (-{PAPER_PARAMS["P_V2G"]} kW)'),
        Patch(facecolor=idle_bar_color, edgecolor='black', alpha=0.8, label='Idle (0 kW)')
    ]
    ax1.legend(handles=legend_elements, loc='upper left', fontsize=12)
    fig.tight_layout()
    plt.savefig(filename)
    plt.close(fig)
    print(f"Grafico salvato: {filename}")
def create_uncertain_profile(p,t,a): n={};f=a/100.0;[n.update({h:p.get((h-t+24)%24,0)*(1+np.random.uniform(-f,f))}) for h in range(24)];return n

def main():
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

    while True:
        choice = input(
            "Quale implementazione vuoi testare?\n"
            " (1) Simple (Deterministico)\n"
            " (2) Faithful (Stocastico, con correzioni)\n"
            " (3) 100% Faithful (Implementazione letterale del paper)\n"
            " (4) Debug Idle (Ricompensa Idle positiva per test)\n"
            "Scelta: "
        )
        if choice == '1':
            env_class_to_test = V2G_Environment_Paper_Simple
            model_path_to_load = MODEL_PATH_SIMPLE
            results_dir = "case_study_results_simple"
            filename_suffix = "simple"
            break
        elif choice == '2':
            env_class_to_test = V2G_Environment_Paper_Faithful
            model_path_to_load = MODEL_PATH_FAITHFUL
            results_dir = "case_study_results_faithful"
            filename_suffix = "faithful"
            break
        elif choice == '3':
            env_class_to_test = V2G_Environment_Paper_100_Percent_Faithful
            model_path_to_load = MODEL_PATH_100_FAITHFUL
            results_dir = "case_study_results_100_faithful"
            filename_suffix = "100_faithful"
            break
        # MODIFICA: Aggiunta la quarta opzione
        elif choice == '4':
            env_class_to_test = V2G_Environment_Paper_Debug_Idle
            model_path_to_load = MODEL_PATH_DEBUG_IDLE
            results_dir = "case_study_results_debug_idle"
            filename_suffix = "debug_idle"
            break
        else:
            print("Scelta non valida. Riprova.")
            
    print(f"\n--- TEST DELL'IMPLEMENTAZIONE: {env_class_to_test.__name__} ---")

    if not os.path.exists(model_path_to_load): 
        print(f"ERRORE: Modello '{model_path_to_load}' non trovato. Esegui prima l'addestramento per questa versione.", file=sys.stderr)
        sys.exit(1)
        
    agent=DQNAgent(device); agent.load(model_path_to_load); agent.e = 0.0
    os.makedirs(results_dir, exist_ok=True)
    
    case_studies = [
        {'id':1,'name':'Case Study 1: Full Charge, Low SoC, High Prices','soc':0.30,'mode':'soc','target':1.0,'prices':PAPER_PRICE_PROFILE,'start':15},
        {'id':2,'name':'Case Study 2: Full Charge, Low SoC, Low Prices','soc':0.30,'mode':'soc','target':1.0,'prices':LOW_PRICE_PROFILE,'start':0},
        {'id':3,'name':'Case Study 3: 2h Charge, High SoC, High Prices','soc':0.90,'mode':'time','duration':120,'prices':PAPER_PRICE_PROFILE,'start':15},
        {'id':4,'name':'Case Study 4: 2h Charge, High SoC, Low Prices','soc':0.90,'mode':'time','duration':120,'prices':LOW_PRICE_PROFILE,'start':0},
        {'id':5,'name':'Case Study 5: 2h Charge, Low SoC, High Prices','soc':0.30,'mode':'time','duration':120,'prices':PAPER_PRICE_PROFILE,'start':15},
        {'id':6,'name':'Case Study 6: 2h Charge, Low SoC, Low Prices','soc':0.30,'mode':'time','duration':120,'prices':LOW_PRICE_PROFILE,'start':0},
        {'id':7,'name':'Case 1 with Uncertainty','soc':0.30,'mode':'soc','target':1.0,'prices':create_uncertain_profile(PAPER_PRICE_PROFILE,3,20),'start':15},
        {'id':8,'name':'Case 5 with Uncertainty','soc':0.30,'mode':'time','duration':120,'prices':create_uncertain_profile(PAPER_PRICE_PROFILE,3,20),'start':15},
    ]
    
    for case in case_studies:
        print(f"\n--- ESECUZIONE: {case['name']} ---")
        env = env_class_to_test(case['prices'])
        env.soc=case['soc'];env.mode=case['mode'];env.target_soc=case.get('target');env.max_duration_minutes=case.get('duration');env.initial_hour=case['start']
        history=run_charging_session(agent,env)
        
        final_filename = f"case_{case['id']}_{filename_suffix}.png"
        full_path = os.path.join(results_dir, final_filename)
        
        plot_and_save_case_study(history,case['prices'],f"{case['name']} ({filename_suffix.replace('_', ' ').capitalize()})", full_path)
        
    print(f"\n--- ESECUZIONE COMPLETATA --- \nGrafici salvati in: '{results_dir}'")

if __name__=="__main__":main()
