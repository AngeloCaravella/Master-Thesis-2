# ==========================================================================================
# REPLICA DELL'ALGORITMO DAL PAPER (v2.2 - CON OPZIONE DEBUG IDLE)
#
# Descrizione:
# Aggiunta una quarta implementazione "Debug Idle" che forza una ricompensa
# positiva per l'inattività, per verificare il plotting e il comportamento strategico.
# ==========================================================================================
import torch, torch.nn as nn, torch.optim as optim
import numpy as np, random, os, sys
from collections import deque

# ... (PARAMETRI E PROFILI DI PREZZO INVARIATI) ...
PAPER_PRICE_PROFILE = {h: p/100.0 for h,p in {0:4.8,1:4.5,2:4.0,3:3.8,4:3.5,5:3.2,6:3.5,7:4.0,8:4.8,9:5.0,10:5.1,11:5.2,12:6.8,13:7.0,14:7.5,15:7.5,16:8.5,17:10.0,18:10.8,19:11.0,20:9.5,21:7.2,22:7.0,23:4.8}.items()}
LOW_PRICE_PROFILE = {h: np.random.uniform(0.04, 0.055) for h in range(24)}
PAPER_PARAMS = {
    'P_CONVENTIONAL':7.0, 'P_FAST':22.0, 'P_V2G':7.0, 
    'BATTERY_CAPACITY':22.0, 'TIME_PERIOD_MINUTES':15, 
    'V2G_ATTENUATION_b':0.5, 
    'IDLE_DISCOUNT_MAX_p':0.5, 'IDLE_DISCOUNT_RATE_d':0.005,
    'DEGRADATION_COST_B_SIMPLE':0.05, 'WAITING_COST_FACTOR_W_SIMPLE':0.01,
    'DEGRADATION_COST_MU': 0.1, 'DEGRADATION_COST_SIGMA': 0.05, 'DEGRADATION_COST_LIMIT': 0.05,
    'IDLE_WAITING_COST_MU': 0.05, 'IDLE_WAITING_COST_SIGMA': 0.0075,
    'CHARGING_WAITING_COST_FACTOR': 0.5,
}
DQN_PARAMS = {'STATE_SIZE':4, 'ACTION_SIZE':4, 'HIDDEN_SIZE':256, 'BUFFER_SIZE':200000, 'BATCH_SIZE':256, 'GAMMA':0.99, 'LEARNING_RATE':0.0005, 'EPSILON_START':1.0, 'EPSILON_END':0.01, 'EPSILON_DECAY':0.9995, 'TARGET_UPDATE_FREQ':10, 'NUM_EPISODES':6000}

MODEL_PATH_SIMPLE = "dqn_model_simple.pth"
MODEL_PATH_FAITHFUL = "dqn_model_faithful.pth"
MODEL_PATH_100_FAITHFUL = "dqn_model_100_faithful.pth"
# MODIFICA: Aggiunto path per il modello di debug
MODEL_PATH_DEBUG_IDLE = "dqn_model_debug_idle.pth"

# ... (CLASSE BASE E ALTRE CLASSI DI AMBIENTE INVARIATE) ...
class V2G_Environment_Base:
    def __init__(self, daily_price_profile: dict):
        self.prices = daily_price_profile
        self.tau = PAPER_PARAMS['TIME_PERIOD_MINUTES'] / 60.0
    def reset(self, initial_soc: float, mode: str, target_soc=None, max_duration_minutes=None, start_hour=0):
        self.current_hour, self.current_minute = start_hour, 0
        self.initial_hour, self.total_idle_steps = self.current_hour, 0
        self.previous_action = 0; self.soc = initial_soc
        self.mode, self.target_soc, self.max_duration_minutes = mode, target_soc, max_duration_minutes
        # Flag per l'ambiente 100% fedele
        if isinstance(self, V2G_Environment_Paper_100_Percent_Faithful):
            self.is_in_forced_charge = False
        return self._get_state()
    def _get_state(self) -> np.ndarray:
        return np.array([self.soc, (self.current_hour%24)/23.0, self.current_minute/45.0, self.previous_action/(DQN_PARAMS['ACTION_SIZE']-1)], dtype=np.float32)
    def _update_time_and_check_done(self, reward):
        self.current_minute += PAPER_PARAMS['TIME_PERIOD_MINUTES']
        if self.current_minute >= 60: self.current_hour += 1; self.current_minute = 0
        done = False
        new_reward = reward
        elapsed_minutes = (self.current_hour * 60 + self.current_minute) - (self.initial_hour * 60)
        if self.soc < 0.2: done = True; new_reward -= 50
        elif self.mode == 'soc' and self.soc >= self.target_soc: done = True; new_reward += 50
        elif self.mode == 'time' and elapsed_minutes >= self.max_duration_minutes: done = True; new_reward += 25
        elif elapsed_minutes >= (24*60 - 1):
            done = True
            if self.mode == 'soc' and self.soc < self.target_soc: new_reward -= 50
        return new_reward, done
class V2G_Environment_Paper_Simple(V2G_Environment_Base):
    def step(self, action: int):
        current_price = self.prices.get(self.current_hour % 24, 0.05)
        waiting_cost = PAPER_PARAMS['WAITING_COST_FACTOR_W_SIMPLE'] * self.tau
        reward = 0
        if action == 0:
            self.total_idle_steps += 1
            idle_hours = self.total_idle_steps * self.tau
            discount = min(PAPER_PARAMS['IDLE_DISCOUNT_MAX_p'], idle_hours * PAPER_PARAMS['IDLE_DISCOUNT_RATE_d'])
            reward = discount - waiting_cost
        elif action == 1:
            energy = PAPER_PARAMS['P_CONVENTIONAL'] * self.tau
            self.soc += energy / PAPER_PARAMS['BATTERY_CAPACITY']
            reward = -(energy * current_price) - waiting_cost
        elif action == 2:
            energy = PAPER_PARAMS['P_FAST'] * self.tau
            self.soc += energy / PAPER_PARAMS['BATTERY_CAPACITY']
            reward = -(energy * current_price) - waiting_cost
        elif action == 3:
            energy = PAPER_PARAMS['P_V2G'] * self.tau
            self.soc -= energy / PAPER_PARAMS['BATTERY_CAPACITY']
            alpha = self.soc ** PAPER_PARAMS['V2G_ATTENUATION_b']
            revenue = alpha * energy * current_price
            reward = revenue - PAPER_PARAMS['DEGRADATION_COST_B_SIMPLE'] - waiting_cost
        self.soc = np.clip(self.soc, 0.0, 1.0)
        self.previous_action = action
        final_reward, done = self._update_time_and_check_done(reward)
        return self._get_state(), final_reward, done
class V2G_Environment_Paper_Faithful(V2G_Environment_Base):
    def step(self, action: int):
        current_price = self.prices.get(self.current_hour % 24, 0.05)
        reward = 0
        idle_waiting_cost = np.random.normal(loc=PAPER_PARAMS['IDLE_WAITING_COST_MU'], scale=PAPER_PARAMS['IDLE_WAITING_COST_SIGMA'])
        charging_waiting_cost = PAPER_PARAMS['CHARGING_WAITING_COST_FACTOR'] * idle_waiting_cost
        if action == 0:
            self.total_idle_steps += 1
            idle_hours = self.total_idle_steps * self.tau
            discount = min(PAPER_PARAMS['IDLE_DISCOUNT_MAX_p'], idle_hours * PAPER_PARAMS['IDLE_DISCOUNT_RATE_d'])
            reward = discount - idle_waiting_cost
        elif action == 1:
            energy = PAPER_PARAMS['P_CONVENTIONAL'] * self.tau
            self.soc += energy / PAPER_PARAMS['BATTERY_CAPACITY']
            reward = -(energy * current_price) - charging_waiting_cost
        elif action == 2:
            energy = PAPER_PARAMS['P_FAST'] * self.tau
            self.soc += energy / PAPER_PARAMS['BATTERY_CAPACITY']
            reward = -(energy * current_price) - charging_waiting_cost
        elif action == 3:
            energy = PAPER_PARAMS['P_V2G'] * self.tau
            self.soc -= energy / PAPER_PARAMS['BATTERY_CAPACITY']
            alpha = self.soc ** PAPER_PARAMS['V2G_ATTENUATION_b']
            revenue = alpha * energy * current_price
            degradation_cost_sample = np.random.normal(loc=PAPER_PARAMS['DEGRADATION_COST_MU'], scale=PAPER_PARAMS['DEGRADATION_COST_SIGMA'])
            degradation_cost = max(degradation_cost_sample, PAPER_PARAMS['DEGRADATION_COST_LIMIT'])
            reward = revenue - degradation_cost - charging_waiting_cost
        self.soc = np.clip(self.soc, 0.0, 1.0)
        self.previous_action = action
        final_reward, done = self._update_time_and_check_done(reward)
        return self._get_state(), final_reward, done
class V2G_Environment_Paper_100_Percent_Faithful(V2G_Environment_Base):
    def step(self, action: int):
        if self.is_in_forced_charge:
            actual_action = 1 
            self.is_in_forced_charge = False
        else:
            actual_action = action
        current_price = self.prices.get(self.current_hour % 24, 0.05)
        reward = 0
        idle_waiting_cost = np.random.normal(loc=PAPER_PARAMS['IDLE_WAITING_COST_MU'], scale=PAPER_PARAMS['IDLE_WAITING_COST_SIGMA'])
        charging_waiting_cost = PAPER_PARAMS['CHARGING_WAITING_COST_FACTOR'] * idle_waiting_cost
        if actual_action == 0:
            self.is_in_forced_charge = True
            self.total_idle_steps += 1
            omega = self.total_idle_steps * self.tau
            d = PAPER_PARAMS['IDLE_DISCOUNT_RATE_d']
            p = PAPER_PARAMS['IDLE_DISCOUNT_MAX_p']
            if 1 - d * omega > p: kappa = 1.0
            elif 1 - d * omega < p: kappa = p
            else: kappa = 1 - d * omega
            costo_ricarica_fittizio = PAPER_PARAMS['P_CONVENTIONAL'] * self.tau * current_price
            reward = -kappa * (costo_ricarica_fittizio + self.tau * charging_waiting_cost + self.tau * idle_waiting_cost)
        elif actual_action == 1:
            energy = PAPER_PARAMS['P_CONVENTIONAL'] * self.tau
            self.soc += energy / PAPER_PARAMS['BATTERY_CAPACITY']
            reward = -(energy * current_price) - charging_waiting_cost
        elif actual_action == 2:
            energy = PAPER_PARAMS['P_FAST'] * self.tau
            self.soc += energy / PAPER_PARAMS['BATTERY_CAPACITY']
            reward = -(energy * current_price) - charging_waiting_cost
        elif actual_action == 3:
            energy = PAPER_PARAMS['P_V2G'] * self.tau
            self.soc -= energy / PAPER_PARAMS['BATTERY_CAPACITY']
            alpha = self.soc ** PAPER_PARAMS['V2G_ATTENUATION_b']
            revenue = alpha * self.tau * PAPER_PARAMS['P_CONVENTIONAL'] * current_price
            degradation_cost_sample = np.random.normal(loc=PAPER_PARAMS['DEGRADATION_COST_MU'], scale=PAPER_PARAMS['DEGRADATION_COST_SIGMA'])
            degradation_cost_base = max(degradation_cost_sample, PAPER_PARAMS['DEGRADATION_COST_LIMIT'])
            degradation_cost = self.tau * PAPER_PARAMS['P_CONVENTIONAL'] * degradation_cost_base
            reward = revenue - degradation_cost - charging_waiting_cost
        self.soc = np.clip(self.soc, 0.0, 1.0)
        self.previous_action = actual_action
        final_reward, done = self._update_time_and_check_done(reward)
        return self._get_state(), final_reward, done

# --- MODIFICA: Quarta classe di ambiente per il debug ---
class V2G_Environment_Paper_Debug_Idle(V2G_Environment_Base):
    """
    Questa classe serve solo a scopo di debug e dimostrazione.
    La ricompensa per l'azione 'Idle' è una costante positiva per incentivare
    l'agente a usarla, dimostrando che il plotting funziona.
    """
    def step(self, action: int) -> tuple[np.ndarray, float, bool]:
        current_price = self.prices.get(self.current_hour % 24, 0.05)
        reward = 0

        if action == 0: # Idle
            # La ricompensa è una costante positiva per renderla attraente
            reward = 0.01
        elif action == 1: # Conventional Charging
            energy = PAPER_PARAMS['P_CONVENTIONAL'] * self.tau
            self.soc += energy / PAPER_PARAMS['BATTERY_CAPACITY']
            reward = -(energy * current_price)
        elif action == 2: # Fast Charging
            energy = PAPER_PARAMS['P_FAST'] * self.tau
            self.soc += energy / PAPER_PARAMS['BATTERY_CAPACITY']
            reward = -(energy * current_price)
        elif action == 3: # V2G
            energy = PAPER_PARAMS['P_V2G'] * self.tau
            self.soc -= energy / PAPER_PARAMS['BATTERY_CAPACITY']
            alpha = self.soc ** PAPER_PARAMS['V2G_ATTENUATION_b']
            revenue = alpha * energy * current_price
            reward = revenue - PAPER_PARAMS['DEGRADATION_COST_B_SIMPLE']
        
        self.soc = np.clip(self.soc, 0.0, 1.0)
        self.previous_action = action
        final_reward, done = self._update_time_and_check_done(reward)
        return self._get_state(), final_reward, done

# ... (CLASSE DQNAgent INVARIATA) ...
class QNetwork(nn.Module):
    def __init__(self,s,a,h):super().__init__();self.net=nn.Sequential(nn.Linear(s,h),nn.ReLU(),nn.Linear(h,h),nn.ReLU(),nn.Linear(h,a))
    def forward(self,x):return self.net(x)
class DQNAgent:
    def __init__(self,d):self.d=d;self.s=DQN_PARAMS['STATE_SIZE'];self.a=DQN_PARAMS['ACTION_SIZE'];self.p=QNetwork(self.s,self.a,DQN_PARAMS['HIDDEN_SIZE']).to(d);self.t=QNetwork(self.s,self.a,DQN_PARAMS['HIDDEN_SIZE']).to(d);self.o=optim.Adam(self.p.parameters(),lr=DQN_PARAMS['LEARNING_RATE']);self.m=deque(maxlen=DQN_PARAMS['BUFFER_SIZE']);self.e=DQN_PARAMS['EPSILON_START'];self.update_target()
    def update_target(self):self.t.load_state_dict(self.p.state_dict())
    def remember(self,s,a,r,n,d):self.m.append((s,a,r,n,d))
    def choose_action(self,s):
        if random.random()<self.e:return random.randrange(self.a)
        with torch.no_grad():return torch.argmax(self.p(torch.FloatTensor(s).unsqueeze(0).to(self.d))).item()
    def learn(self):
        if len(self.m)<DQN_PARAMS['BATCH_SIZE']:return
        b=random.sample(self.m,DQN_PARAMS['BATCH_SIZE']);s,a,r,n,d=zip(*b);s=torch.FloatTensor(np.array(s)).to(self.d);a=torch.LongTensor(a).unsqueeze(1).to(self.d);r=torch.FloatTensor(r).unsqueeze(1).to(self.d);n=torch.FloatTensor(np.array(n)).to(self.d);d=torch.BoolTensor(d).unsqueeze(1).to(self.d);c=self.p(s).gather(1,a)
        with torch.no_grad():t=r+(DQN_PARAMS['GAMMA']*self.t(n).max(1)[0].unsqueeze(1)*(~d))
        l=nn.MSELoss()(c,t);self.o.zero_grad();l.backward();self.o.step()
    def decay_epsilon(self):
        if self.e>DQN_PARAMS['EPSILON_END']:self.e*=DQN_PARAMS['EPSILON_DECAY']
    def save(self,p):torch.save(self.p.state_dict(),p);print(f"\nModello salvato: {p}")
    def load(self,p):self.p.load_state_dict(torch.load(p,map_location=self.d));self.update_target();print(f"Modello caricato: {p}")

def run_training(agent, env_class, model_path_to_save):
    print(f"\n--- AVVIO ADDESTRAMENTO PER L'AMBIENTE: {env_class.__name__} ---")
    rewards_history=[]
    env_high = env_class(PAPER_PRICE_PROFILE)
    env_low = env_class(LOW_PRICE_PROFILE)
    for episode in range(DQN_PARAMS['NUM_EPISODES']):
        env,start_hour=(env_high,15) if random.random()>0.5 else (env_low,0)
        mode,target_soc,duration=('soc',1.0,None) if random.random()>0.5 else ('time',None,random.choice([120,240,360]))
        state=env.reset(random.uniform(0.2,0.9),mode,target_soc,duration,start_hour=start_hour)
        done,total_reward=False,0
        for _ in range(96*2):
            action=agent.choose_action(state);next_state,reward,done=env.step(action)
            agent.remember(state,action,reward,next_state,done);agent.learn();state=next_state;total_reward+=reward
            if done:break
        agent.decay_epsilon();rewards_history.append(total_reward)
        if (episode+1)%DQN_PARAMS['TARGET_UPDATE_FREQ']==0:agent.update_target()
        if (episode+1)%200==0:print(f"Episodio {episode+1}/{DQN_PARAMS['NUM_EPISODES']}, Avg Reward: {np.mean(rewards_history[-200:]):.2f}, Epsilon: {agent.e:.3f}")
    agent.save(model_path_to_save)

def main():
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu");agent=DQNAgent(device)
    
    while True:
        choice = input(
            "Quale modello vuoi addestrare?\n"
            " (1) Simple (Deterministico)\n"
            " (2) Faithful (Stocastico, con correzioni)\n"
            " (3) 100% Faithful (Implementazione letterale del paper)\n"
            " (4) Debug Idle (Ricompensa Idle positiva per test)\n"
            "Scelta: "
        )
        if choice == '1':
            env_class_to_train = V2G_Environment_Paper_Simple
            model_path = MODEL_PATH_SIMPLE
            break
        elif choice == '2':
            env_class_to_train = V2G_Environment_Paper_Faithful
            model_path = MODEL_PATH_FAITHFUL
            break
        elif choice == '3':
            env_class_to_train = V2G_Environment_Paper_100_Percent_Faithful
            model_path = MODEL_PATH_100_FAITHFUL
            break
        # MODIFICA: Aggiunta la quarta opzione
        elif choice == '4':
            env_class_to_train = V2G_Environment_Paper_Debug_Idle
            model_path = MODEL_PATH_DEBUG_IDLE
            break
        else:
            print("Scelta non valida. Riprova.")

    if os.path.exists(model_path):
        if input(f"Modello '{model_path}' trovato. (1) Ri-addestra, (2) Esci: ")!='1':return
    
    run_training(agent, env_class_to_train, model_path)
    print(f"\nAddestramento completato. Esegui 'Paper.py' per testare i risultati.")

if __name__=="__main__":main()
