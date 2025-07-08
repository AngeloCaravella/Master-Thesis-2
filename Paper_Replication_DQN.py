# ==========================================================================================
# REPLICA DELL'ALGORITMO DAL PAPER (v1.6 - REPLICA FEDELE DELLA RICOMPENSA)
#
# Descrizione:
# Versione definitiva che implementa la funzione di ricompensa quasi 1:1
# rispetto alle equazioni (5)-(10) del paper, inclusa la logica di sconto per
# l'inattività (Idle). Questo è cruciale per apprendere strategie complesse.
# ==========================================================================================
import torch, torch.nn as nn, torch.optim as optim
import numpy as np, random, os, sys
from collections import deque

# --- PROFILI DI PREZZO FEDELI AL PAPER ---
PAPER_PRICE_PROFILE = {h: p/100.0 for h,p in {0:4.8,1:4.5,2:4.0,3:3.8,4:3.5,5:3.2,6:3.5,7:4.0,8:4.8,9:5.0,10:5.1,11:5.2,12:6.8,13:7.0,14:7.5,15:7.5,16:8.5,17:10.0,18:10.8,19:11.0,20:9.5,21:7.2,22:7.0,23:4.8}.items()}
LOW_PRICE_PROFILE = {h: np.random.uniform(0.04, 0.055) for h in range(24)}

# Parametri e costanti
PAPER_PARAMS = {'P_CONVENTIONAL':7.0, 'P_FAST':22.0, 'P_V2G':7.0, 'BATTERY_CAPACITY':22.0, 'TIME_PERIOD_MINUTES':15, 'V2G_ATTENUATION_b':0.5, 'DEGRADATION_COST_B':0.05, 'IDLE_DISCOUNT_MAX_p':0.5, 'IDLE_DISCOUNT_RATE_d':0.005, 'WAITING_COST_FACTOR_W':0.01}
DQN_PARAMS = {'STATE_SIZE':4, 'ACTION_SIZE':4, 'HIDDEN_SIZE':256, 'BUFFER_SIZE':200000, 'BATCH_SIZE':256, 'GAMMA':0.99, 'LEARNING_RATE':0.0005, 'EPSILON_START':1.0, 'EPSILON_END':0.01, 'EPSILON_DECAY':0.9995, 'TARGET_UPDATE_FREQ':10, 'NUM_EPISODES':6000}
MODEL_PATH = "dqn_model_paper_true_replication.pth"

class V2G_Environment_Paper_Goal_Oriented:
    def __init__(self, daily_price_profile: dict):
        self.prices = daily_price_profile
        self.tau = PAPER_PARAMS['TIME_PERIOD_MINUTES'] / 60.0

    def reset(self, initial_soc: float, mode: str, target_soc=None, max_duration_minutes=None, start_hour=0):
        self.current_hour, self.current_minute = start_hour, 0
        self.initial_hour, self.total_idle_steps = self.current_hour, 0
        self.previous_action = 0; self.soc = initial_soc
        self.mode, self.target_soc, self.max_duration_minutes = mode, target_soc, max_duration_minutes
        return self._get_state()

    def _get_state(self) -> np.ndarray:
        return np.array([self.soc, (self.current_hour%24)/23.0, self.current_minute/45.0, self.previous_action/(DQN_PARAMS['ACTION_SIZE']-1)], dtype=np.float32)

    def step(self, action: int) -> tuple[np.ndarray, float, bool]:
        current_price = self.prices.get(self.current_hour % 24, 0.05)
        waiting_cost = PAPER_PARAMS['WAITING_COST_FACTOR_W'] * self.tau
        reward = 0

        if action == 0: # Idle
            self.total_idle_steps += 1
            idle_hours = self.total_idle_steps * self.tau
            discount = min(PAPER_PARAMS['IDLE_DISCOUNT_MAX_p'], idle_hours * PAPER_PARAMS['IDLE_DISCOUNT_RATE_d'])
            reward = discount - waiting_cost
        elif action == 1: # Conventional Charging
            energy = PAPER_PARAMS['P_CONVENTIONAL'] * self.tau
            self.soc += energy / PAPER_PARAMS['BATTERY_CAPACITY']
            reward = -(energy * current_price) - waiting_cost
        elif action == 2: # Fast Charging
            energy = PAPER_PARAMS['P_FAST'] * self.tau
            self.soc += energy / PAPER_PARAMS['BATTERY_CAPACITY']
            reward = -(energy * current_price) - waiting_cost
        elif action == 3: # V2G
            energy = PAPER_PARAMS['P_V2G'] * self.tau
            self.soc -= energy / PAPER_PARAMS['BATTERY_CAPACITY']
            alpha = self.soc ** PAPER_PARAMS['V2G_ATTENUATION_b']
            revenue = alpha * energy * current_price
            reward = revenue - PAPER_PARAMS['DEGRADATION_COST_B'] - waiting_cost
        
        self.soc = np.clip(self.soc, 0.0, 1.0)
        self.previous_action = action
        self.current_minute += PAPER_PARAMS['TIME_PERIOD_MINUTES']
        if self.current_minute >= 60: self.current_hour += 1; self.current_minute = 0

        done = False
        elapsed_minutes = (self.current_hour * 60 + self.current_minute) - (self.initial_hour * 60)
        
        if self.soc < 0.2: done = True; reward -= 50
        elif self.mode == 'soc' and self.soc >= self.target_soc: done = True; reward += 50
        elif self.mode == 'time' and elapsed_minutes >= self.max_duration_minutes: done = True; reward += 25
        elif elapsed_minutes >= (24*60 - 1):
            done = True
            if self.mode == 'soc' and self.soc < self.target_soc: reward -= 50

        return self._get_state(), reward, done

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

def run_training(agent):
    print("\n--- AVVIO ADDESTRAMENTO (CON LOGICA DI RICOMPENSA FEDELE) ---")
    rewards_history=[]
    env_high=V2G_Environment_Paper_Goal_Oriented(PAPER_PRICE_PROFILE);env_low=V2G_Environment_Paper_Goal_Oriented(LOW_PRICE_PROFILE)
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
    agent.save(MODEL_PATH)

def main():
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu");agent=DQNAgent(device)
    if os.path.exists(MODEL_PATH):
        if input(f"Modello '{MODEL_PATH}' trovato. (1) Ri-addestra, (2) Esci: ")!='1':return
    run_training(agent);print(f"\nAddestramento completato. Esegui 'Paper.py' per i risultati.")

if __name__=="__main__":main()
