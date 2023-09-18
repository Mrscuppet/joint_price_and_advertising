import numpy as np
import random


def exp_ctr(time,advertisers):

    exp_ctr = ((time/advertisers) + advertisers)/(time+advertisers)
    return exp_ctr

def exp_ctr_plus(time,advertisers): #da modifcare coem sopra se lo voglio usare

    exp_ctr = ((time/advertisers) + advertisers)/(time+advertisers)*1.20
    return exp_ctr

class UCB1Agent():
    def __init__(self, n_arms, max_reward):
        self.n_arms = n_arms
        self.max_reward = max_reward                                    #maxreward è un vettore che indica la differenza per ogni arm tra price e bid.
        self.arms = np.arange(self.n_arms)
        self.t = 0
        self.a_hist = []
        self.last_pull = None
        self.reset()

    def reset(self):
        self.t = 0
        self.last_pull = None
        self.avg_reward = np.array([np.inf for _ in range(self.n_arms)])
        self.rewards = [[] for _ in range(self.n_arms)]
        self.n_pulls = np.zeros(self.n_arms)
        return self

    def pull_arm(self):
        ucb1 = [self.avg_reward[a]+self.max_reward[a] *
                np.sqrt(2*np.log(self.t)/self.n_pulls[a]) for a in range(self.n_arms)]
        #print(ucb1)
        self.last_pull = np.argmax(ucb1)
        new_a = self.arms[self.last_pull]
        self.n_pulls[self.last_pull] += 1
        #print("Il numero di pulls dell arm",self.last_pull,"è: ",self.n_pulls[self.last_pull])
        self.a_hist.append(new_a)
        return new_a

    def update(self, C):  
        #print("La rewards sono:",self.rewards[self.last_pull],"con aggiunta",C)                               #C è la reward ottenuta dal mab nell'ultimo step.
        self.rewards[self.last_pull].append(C)
        if self.n_pulls[self.last_pull] == 1:
            self.avg_reward[self.last_pull] = C
            #print("primo",self.avg_reward[self.last_pull])
        else:
            self.avg_reward[self.last_pull] = (
                self.avg_reward[self.last_pull]*(self.n_pulls[self.last_pull]-1)+C)/(self.n_pulls[self.last_pull])
            #print("non primo",self.avg_reward[self.last_pull])

        self.t += 1

class UCB1_Sliding_Window_Agent():
    def __init__(self, n_arms, max_reward,tau):
        self.n_arms = n_arms
        self.max_reward = max_reward                                    #maxreward è un vettore che indica la differenza per ogni arm tra price e bid.
        self.arms = np.arange(self.n_arms)
        self.t = 0
        self.a_hist = []
        self.last_pull = None
        self.tau = tau
        self.reset()

    def reset(self):
        self.t = 0
        self.last_pull = None
        self.avg_reward = np.array([np.inf for _ in range(self.n_arms)])
        self.rewards = [[] for _ in range(self.n_arms)]
        self.n_pulls = np.zeros(self.n_arms)
        self.list_n_pulls = [[] for _ in range(self.n_arms)]
        self.rewards_tau =  [[] for _ in range(self.n_arms)] #questo è stato aggiunto per avere un vettore delle reward anche quando il braccio non è stato sleezionato per calcolare correttamente le rewards di un arm nell arco Tau, altrimenti non si avrebbe conoscenza dle tempo t e quindi la somma verrebbe male  quando si fa l update dell avg reward.

        return self

    def pull_arm_window(self):
        if self.t <= self.tau: 
            ucb1 = [self.avg_reward[a]+self.max_reward[a] *
                    np.sqrt(2*np.log(self.t)/self.n_pulls[a]) for a in range(self.n_arms)]
            #print("Questo è lo ucb:",ucb1)
            self.last_pull = np.argmax(ucb1)
            new_a = self.arms[self.last_pull]
            self.n_pulls[self.last_pull] += 1
            self.list_n_pulls[self.last_pull].append(1)
            for pull in range(len(self.list_n_pulls)):      #ho aggiunto questo for per aggiungere uno 0 agli altri vettori se non sono stati pullati, mi serve per conteggiare il giusto numero di volte in cui un arm è stato tirato le ultime Tau volte
                if pull != self.last_pull:
                    self.list_n_pulls[pull].append(0)

            #print("Lista di pulls",self.list_n_pulls,"\n")

            self.a_hist.append(new_a)
            return new_a
        else:
            #print("ecco avg re",self.avg_reward)
            #print("ecco le max rew",self.max_reward)
            #print("questo è il time",self.t)
            #print("questo è tau",self.tau)
            #print("questo è lista dei pulls",self.list_n_pulls)
            #for a in range(self.n_arms):
                #print("ecco le somme",sum(self.list_n_pulls[a][(self.t - self.tau):]))
            ucb1 = [self.avg_reward[a]+self.max_reward[a] *
                    np.sqrt(2*np.log(self.tau)/sum(self.list_n_pulls[a][(self.t - self.tau):])) for a in range(self.n_arms)]
            self.last_pull = np.argmax(ucb1)
            new_a = self.arms[self.last_pull]
            #print("ecco ucb1",ucb1)
            #print("ARM SLEEZIONATO",new_a)
            self.n_pulls[self.last_pull] += 1
            self.list_n_pulls[self.last_pull].append(1)
            for pull in range(len(self.list_n_pulls)):      #ho aggiunto questo for per aggiungere uno 0 agli altri vettori se non sono stati pullati, mi serve per conteggiare il giusto numero di volte in cui un arm è stato tirato le ultime Tau volte
                if pull != self.last_pull:
                    #print("weeee",self.list_n_pulls)

                    self.list_n_pulls[pull].append(0)

            #print("Lista di pulls",self.list_n_pulls,"\n")

            self.a_hist.append(new_a)
            return new_a

    def update(self, C):                                                 #C è la reward ottenuta dal mab nell'ultimo step.
        self.rewards[self.last_pull].append(C)
        self.rewards_tau[self.last_pull].append(C)
        
        for pull in range(len(self.list_n_pulls)):
            if pull != self.last_pull:
                self.rewards_tau[pull].append(0)

       #print("Lista rewards:",self.rewards_tau,"\n")
        if self.t <= self.tau:
            if self.n_pulls[self.last_pull] == 1:
                self.avg_reward[self.last_pull] = C
            else:
                self.avg_reward[self.last_pull] = (
                    self.avg_reward[self.last_pull]*(self.n_pulls[self.last_pull]-1)+C)/(self.n_pulls[self.last_pull])
            self.t += 1
        else:
            self.avg_reward[self.last_pull] = (sum(self.rewards_tau[self.last_pull][(self.t - self.tau):]))/sum(self.list_n_pulls[self.last_pull][self.t-self.tau:])
            self.t += 1

        #print("rewards_tau",self.rewards_tau)
        #print("Ecco average reward",self.avg_reward)

class UCB1_Variable_Sliding_Window_Variation_Agent():
    def __init__(self, n_arms, max_reward,tau,tau_max,tau_min,tau_step,m1,m0):
        self.n_arms = n_arms
        self.max_reward = max_reward                                    #maxreward è un vettore che indica la differenza per ogni arm tra price e bid.
        self.arms = np.arange(self.n_arms)
        self.t = 0
        self.a_hist = []
        self.last_pull = None
        self.tau = tau
        self.tau_max = tau_max
        self.tau_min = tau_min
        self.tau_step = tau_step
        self.m1 = m1
        self.m0 = m0
        self.reset()

    def reset(self):
        self.t = 0
        self.last_pull = None
        self.avg_reward = np.array([np.inf for _ in range(self.n_arms)])
        self.rewards = [[] for _ in range(self.n_arms)]
        self.n_pulls = np.zeros(self.n_arms)
        self.list_n_pulls = [[] for _ in range(self.n_arms)]
        self.rewards_tau =  [[] for _ in range(self.n_arms)] #questo è stato aggiunto per avere un vettore delle reward anche quando il braccio non è stato sleezionato per calcolare correttamente le rewards di un arm nell arco Tau, altrimenti non si avrebbe conoscenza dle tempo t e quindi la somma verrebbe male  quando si fa l update dell avg reward.
        self.count = []


        return self

    def pull_arm_window(self):

        if self.t <= self.tau: 
            ucb1 = [self.avg_reward[a]+self.max_reward[a] *
                    np.sqrt(2*np.log(self.t)/self.n_pulls[a]) for a in range(self.n_arms)]
            #print("Questo è lo ucb:",ucb1)
            self.last_pull = np.argmax(ucb1)
            new_a = self.arms[self.last_pull]
            self.n_pulls[self.last_pull] += 1
            self.list_n_pulls[self.last_pull].append(1)
            for pull in range(len(self.list_n_pulls)):      #ho aggiunto questo for per aggiungere uno 0 agli altri vettori se non sono stati pullati, mi serve per conteggiare il giusto numero di volte in cui un arm è stato tirato le ultime Tau volte
                if pull != self.last_pull:
                    self.list_n_pulls[pull].append(0)

            #print("Lista di pulls",self.list_n_pulls,"\n")

            self.a_hist.append(new_a)
            return new_a
        else:

            #print("ecco avg re",self.avg_reward)
            #print("ecco le max rew",self.max_reward)
            #print("questo è il time",self.t)
            #print("questo è tau",self.tau)
            #print("questo è lista dei pulls",self.list_n_pulls)
            #for a in range(self.n_arms):
                #print("ecco le somme",sum(self.list_n_pulls[a][(self.t - self.tau):]))
            ucb1 = [self.avg_reward[a]+self.max_reward[a] *
                    np.sqrt(2*np.log(self.tau)/sum(self.list_n_pulls[a][(self.t - self.tau):])) for a in range(self.n_arms)]
            self.last_pull = np.argmax(ucb1)
            new_a = self.arms[self.last_pull]
            #print("ecco ucb1",ucb1)
            #print("ARM SLEEZIONATO",new_a)
            self.n_pulls[self.last_pull] += 1
            self.list_n_pulls[self.last_pull].append(1)
            for pull in range(len(self.list_n_pulls)):      #ho aggiunto questo for per aggiungere uno 0 agli altri vettori se non sono stati pullati, mi serve per conteggiare il giusto numero di volte in cui un arm è stato tirato le ultime Tau volte
                if pull != self.last_pull:
                    #print("weeee",self.list_n_pulls)

                    self.list_n_pulls[pull].append(0)

            #print("Lista di pulls",self.list_n_pulls,"\n")

            self.a_hist.append(new_a)
            return new_a

    def update(self, C):                                                 #C è la reward ottenuta dal mab nell'ultimo step.
        self.rewards[self.last_pull].append(C)
        self.rewards_tau[self.last_pull].append(C)

        self.count.append(np.sign(C-self.avg_reward[self.last_pull]))
        #print("Ecco il sign delle reward",self.count)

        if  self.t > self.tau + self.tau_step: #check in modo che non ci sia un update di tau che sfori i samples

            #print("Time",self.t)
            #print("Difference",self.t-self.m1)
            #print("Count di 1",self.count[(self.t-self.m1):].count(1))
            if self.count[(self.t-self.m1):].count(1) == self.m1:    #check se gli ultimi m1 elementi di count son tutti 1 ( quindi segno positivo)

                #print("Ciao sono tutti positivi")

                if self.tau > self.tau_min:

                    self.tau = self.tau - self.tau_step
             
                    print("Tau aggiornato positivi",self.tau)
            #print("time",self.t)
            #print("counting 0",self.count[(self.t-self.m0+1):].count(0) == 0)
            #print("countin pari",(set(self.count[(self.t-self.m0+1)::2])))
            #print("counting dispari",(set(self.count[self.t-self.m0+1+1::2])))
            if (self.count[(self.t-self.m0+1):].count(0) == 0) and self.count[(self.t-self.m1+1):].count(-1) != self.m1 and ((len(set(self.count[(self.t-self.m0+1)::2])) == 1) and (len(set(self.count[self.t-self.m0+1+1::2])) == 1)) == True: #check se non c'è nemmeno uno 0, e se gli ultimi m0 elementi sono un alternanza di due valori (dato che abbiamo escluso lo 0 rimangono solo -1 e 1)
                
                #print("Ciao sono tutti alternati")
                #print("Time è: ",self.t)

                if self.tau < self.tau_max:

                    self.tau = self.tau + self.tau_step

                    print("Tau aggiornato alternato",self.tau)
        
        for pull in range(len(self.list_n_pulls)):
            if pull != self.last_pull:
                self.rewards_tau[pull].append(0)

       #print("Lista rewards:",self.rewards_tau,"\n")
        if self.t <= self.tau:
            if self.n_pulls[self.last_pull] == 1:
                self.avg_reward[self.last_pull] = C
            else:
                self.avg_reward[self.last_pull] = (
                    self.avg_reward[self.last_pull]*(self.n_pulls[self.last_pull]-1)+C)/(self.n_pulls[self.last_pull])
            self.t += 1
        else:
            self.avg_reward[self.last_pull] = (sum(self.rewards_tau[self.last_pull][(self.t - self.tau):]))/sum(self.list_n_pulls[self.last_pull][self.t-self.tau:])
            self.t += 1

        #print("rewards_tau",self.rewards_tau)
        #print("Ecco average reward",self.avg_reward)


class UCB1_Discounted_Agent():
    def __init__(self, n_arms, max_reward,gamma):
        self.n_arms = n_arms
        self.max_reward = max_reward                                    #maxreward è un vettore che indica la differenza per ogni arm tra price e bid.
        self.arms = np.arange(self.n_arms)
        self.t = 0
        self.a_hist = []
        self.last_pull = None
        self.gamma = gamma
        self.reset()

    def reset(self):
        self.t = 0
        self.last_pull = None
        self.avg_reward = np.array([np.inf for _ in range(self.n_arms)])
        self.rewards = [[] for _ in range(self.n_arms)]
        self.n_pulls = np.zeros(self.n_arms)
        self.list_n_pulls = [[] for _ in range(self.n_arms)]
        self.rewards_gamma =  [[] for _ in range(self.n_arms)] #questo è stato aggiunto per avere un vettore delle reward anche quando il braccio non è stato sleezionato per calcolare correttamente le rewards di un arm nell arco Tau, altrimenti non si avrebbe conoscenza dle tempo t e quindi la somma verrebbe male  quando si fa l update dell avg reward.

        return self

    def pull_arm_window(self):
        ucb1 = [self.avg_reward[a]+self.max_reward[a] *
                np.sqrt(2*np.log(self.t)/self.n_pulls[a]) for a in range(self.n_arms)]

        self.last_pull = np.argmax(ucb1)
        new_a = self.arms[self.last_pull]
        self.n_pulls[self.last_pull] += 1
        self.list_n_pulls[self.last_pull].append(1)
        for pull in range(len(self.list_n_pulls)):      #ho aggiunto questo for per aggiungere uno 0 agli altri vettori se non sono stati pullati, mi serve per conteggiare il giusto numero di volte in cui un arm è stato tirato le ultime Tau volte
            if pull != self.last_pull:
                self.list_n_pulls[pull].append(0)


        self.a_hist.append(new_a)
        return new_a

    def update(self, C):                                                 #C è la reward ottenuta dal mab nell'ultimo step.
        self.rewards[self.last_pull].append(C)
        self.rewards_gamma[self.last_pull].append(C)
        
        for pull in range(len(self.list_n_pulls)):
            if pull != self.last_pull:
                self.rewards_gamma[pull].append(0)

        if self.n_pulls[self.last_pull] == 1:
            self.avg_reward[self.last_pull] = C
        else:

            self.avg_reward[self.last_pull] = (
                sum(self.rewards_gamma[self.last_pull][a]*self.gamma**(self.t-a) for a in range(self.t+1))/sum(self.list_n_pulls[self.last_pull][a]*self.gamma**(self.t-a) for a in range(self.t+1)) )
        self.t += 1


class UCB1_Variable_discounted__Variation_Agent():
    def __init__(self, n_arms, max_reward,gamma,gamma_max,gamma_min,gamma_step,m1,m0):
        self.n_arms = n_arms
        self.max_reward = max_reward                                    #maxreward è un vettore che indica la differenza per ogni arm tra price e bid.
        self.arms = np.arange(self.n_arms)
        self.t = 0
        self.a_hist = []
        self.last_pull = None
        self.gamma = gamma
        self.gamma_max = gamma_max
        self.gamma_min = gamma_min
        self.gamma_step = gamma_step
        self.m1 = m1
        self.m0 = m0
        self.reset()

    def reset(self):
        self.t = 0
        self.last_pull = None
        self.avg_reward = np.array([np.inf for _ in range(self.n_arms)])
        self.rewards = [[] for _ in range(self.n_arms)]
        self.n_pulls = np.zeros(self.n_arms)
        self.list_n_pulls = [[] for _ in range(self.n_arms)]
        self.rewards_gamma =  [[] for _ in range(self.n_arms)] #questo è stato aggiunto per avere un vettore delle reward anche quando il braccio non è stato sleezionato per calcolare correttamente le rewards di un arm nell arco Tau, altrimenti non si avrebbe conoscenza dle tempo t e quindi la somma verrebbe male  quando si fa l update dell avg reward.
        self.count = []

        return self
    
    def pull_arm_window(self):
        ucb1 = [self.avg_reward[a]+self.max_reward[a] *
                np.sqrt(2*np.log(self.t)/self.n_pulls[a]) for a in range(self.n_arms)]

        self.last_pull = np.argmax(ucb1)
        new_a = self.arms[self.last_pull]
        self.n_pulls[self.last_pull] += 1
        self.list_n_pulls[self.last_pull].append(1)
        for pull in range(len(self.list_n_pulls)):      #ho aggiunto questo for per aggiungere uno 0 agli altri vettori se non sono stati pullati, mi serve per conteggiare il giusto numero di volte in cui un arm è stato tirato le ultime Tau volte
            if pull != self.last_pull:
                self.list_n_pulls[pull].append(0)


        self.a_hist.append(new_a)
        return new_a


    def update(self, C):                                                 #C è la reward ottenuta dal mab nell'ultimo step.
        self.rewards[self.last_pull].append(C)
        self.rewards_gamma[self.last_pull].append(C)

        self.count.append(np.sign(C-self.avg_reward[self.last_pull]))



        #print("Time",self.t)
        #print("Difference",self.t-self.m1)
        #print("Count di 1",self.count[(self.t-self.m1):].count(1))
        if self.count[(self.t-self.m1+1):].count(1) == self.m1:    #check se gli ultimi m1 elementi di count son tutti 1 ( quindi segno positivo)

            #print("Ciao sono tutti positivi")

            if self.gamma > self.gamma_min:

                self.gamma = self.gamma - self.gamma_step
            
                print("Tau aggiornato positivi",self.gamma)
        #print("time",self.t)
        #print("counting 0",self.count[(self.t-self.m0+1):].count(0) == 0)
        #print("countin pari",(set(self.count[(self.t-self.m0+1)::2])))
        #print("counting dispari",(set(self.count[self.t-self.m0+1+1::2])))
        if (self.count[(self.t-self.m0+1):].count(0) == 0) and self.count[(self.t-self.m1+1):].count(-1) != self.m1 and ((len(set(self.count[(self.t-self.m0+1)::2])) == 1) and (len(set(self.count[self.t-self.m0+1+1::2])) == 1)) == True: #check se non c'è nemmeno uno 0, e se gli ultimi m0 elementi sono un alternanza di due valori (dato che abbiamo escluso lo 0 rimangono solo -1 e 1)
            
            #print("Ciao sono tutti alternati")

            if self.gamma < self.gamma_max:

                self.gamma = self.gamma + self.gamma_step

                print("Tau aggiornato alternato",self.gamma)
        
        for pull in range(len(self.list_n_pulls)):
            if pull != self.last_pull:
                self.rewards_gamma[pull].append(0)

        
        for pull in range(len(self.list_n_pulls)):
            if pull != self.last_pull:
                self.rewards_gamma[pull].append(0)

        if self.n_pulls[self.last_pull] == 1:
            self.avg_reward[self.last_pull] = C
        else:

            self.avg_reward[self.last_pull] = (
                sum(self.rewards_gamma[self.last_pull][a]*self.gamma**(self.t-a) for a in range(self.t+1))/sum(self.list_n_pulls[self.last_pull][a]*self.gamma**(self.t-a) for a in range(self.t+1)) )
        self.t += 1

    

 
class Exp3Agent():
    def __init__(self, n_arms, max_reward, gamma = 1):
        self.gamma = gamma
        self.max_reward = max_reward
        self.n_arms = n_arms
        self.arms = np.arange(self.n_arms)
        self.t = 0
        self.a_hist = []
        self.last_pull = None
        self.reset()

    def reset(self):
        self.t = 0
        self.last_pull = None        
        self.w = np.ones(self.n_arms)
        self.est_rewards = np.zeros(self.n_arms)
        self.probabilities = (1/self.n_arms)*np.ones(self.n_arms)
        self.probabilities[0] = 1 - sum(self.probabilities[1:])
        return self

    def pull_arm(self):
        new_a = np.random.choice(self.arms, p=self.probabilities, size=None)
        self.last_pull = new_a
        self.a_hist.append(new_a)
        return new_a

    def update(self, C):   
        C[0] = C[0]/self.max_reward[C[1]-50]
        self.est_rewards[self.last_pull] = C[0]/self.probabilities[self.last_pull]
        self.w[self.last_pull] *= np.exp(self.gamma *
                                         self.est_rewards[self.last_pull]/self.n_arms)
        self.w[~np.isfinite(self.w)] = 0
        self.probabilities = (1-self.gamma)*self.w / \
            sum(self.w)+self.gamma/self.n_arms
        self.probabilities[0] = 1 - sum(self.probabilities[1:])


class UCB1Agent_adaptive():
    def __init__(self, n_arms, max_reward,boost):
        self.n_arms = n_arms
        self.max_reward = max_reward                                    #maxreward è un vettore che indica la differenza per ogni arm tra price e bid.
        self.arms = np.arange(self.n_arms)
        self.t = 0
        self.a_hist = []
        self.last_pull = None
        self.boost = boost
        self.reset()

    def reset(self):
        self.t = 0
        self.last_pull = None
        self.avg_reward = np.array([np.inf for _ in range(self.n_arms)])
        self.avg_reward_real = np.array([np.inf for _ in range(self.n_arms)])
        self.rewards = [[] for _ in range(self.n_arms)]
        self.rewards_real = [[] for _ in range(self.n_arms)]
        self.n_pulls = np.zeros(self.n_arms)
        return self

    def pull_arm(self):
        ucb1 = [self.avg_reward[a]+self.max_reward[a] *
                np.sqrt(2*np.log(self.t)/self.n_pulls[a]) for a in range(self.n_arms)]
        #print(ucb1)
        self.last_pull = np.argmax(ucb1)
        new_a = self.arms[self.last_pull]
        self.n_pulls[self.last_pull] += 1
        #print("Il numero di pulls dell arm",self.last_pull,"è: ",self.n_pulls[self.last_pull])
        self.a_hist.append(new_a)
        return new_a

    def update(self,C,CTR,advertisers):  #CTR è il CTR ottenuto fino a quel momento dall'esecuzione
        #print("La rewards sono:",self.rewards[self.last_pull],"con aggiunta",C)                               #C è la reward ottenuta dal mab nell'ultimo step.
        

        if CTR <= exp_ctr_plus(self.t+1,advertisers):
            #print("Il time è ",self.t,"e il ctr è ",CTR,"mentre l'expected è: ",exp_ctr(self.t+1,advertisers),"entro nel 1")

            self.rewards[self.last_pull].append(C*self.boost)
            if self.n_pulls[self.last_pull] == 1:
                self.avg_reward[self.last_pull] = C*self.boost 
            else:
                self.avg_reward[self.last_pull] = (
                    self.avg_reward[self.last_pull]*(self.n_pulls[self.last_pull]-1)+C*self.boost)/(self.n_pulls[self.last_pull])


        else:
            #print("Entro nel 2")
            #print("Il time è ",self.t,"e il ctr è ",CTR,"mentre l'expected è: ",exp_ctr(self.t+1,advertisers),"entro nel 2","l'arm tirato è",self.last_pull,"con lunghezza:",len(self.rewards[self.last_pull]))


            self.rewards[self.last_pull].append(C)
            if self.n_pulls[self.last_pull] == 1:
                self.avg_reward[self.last_pull] = C
                #print("primo",self.avg_reward[self.last_pull])
            else:
                self.avg_reward[self.last_pull] = (
                    self.avg_reward[self.last_pull]*(self.n_pulls[self.last_pull]-1)+C)/(self.n_pulls[self.last_pull])
                #print("non primo",self.avg_reward[self.last_pull])

        self.t += 1

    def update_real(self, C):  
        #print("La rewards sono:",self.rewards[self.last_pull],"con aggiunta",C)                               #C è la reward ottenuta dal mab nell'ultimo step.
        self.rewards_real[self.last_pull].append(C)
        if self.n_pulls[self.last_pull] == 1:
            self.avg_reward_real[self.last_pull] = C
            #print("primo",self.avg_reward[self.last_pull])
        else:
            self.avg_reward_real[self.last_pull] = (
                self.avg_reward_real[self.last_pull]*(self.n_pulls[self.last_pull]-1)+C)/(self.n_pulls[self.last_pull])
            #print("non primo",self.avg_reward[self.last_pull])
    


class UCB1_CTR():
    def __init__(self, n_arms, max_reward):
        self.n_arms = n_arms
        self.max_reward = max_reward                                    #maxreward è un vettore che indica la differenza per ogni arm tra price e bid.
        self.arms = np.arange(self.n_arms)
        self.t = 0
        self.a_hist = []
        self.last_pull = None
        self.reset()

    def reset(self):
        self.t = 0
        self.last_pull = None
        self.avg_reward = np.array([np.inf for _ in range(self.n_arms)])
        self.rewards = [[] for _ in range(self.n_arms)]
        self.n_pulls = np.zeros(self.n_arms)
        self.list_n_pulls = [[] for _ in range(self.n_arms)]

        return self

    def pull_arm(self,CTR,advertisers):

        tau = 5000
        
        #print("Il time è: ",self.t)
        #print("CTR è :",CTR)


        if self.t > tau and CTR < exp_ctr(self.t+1,advertisers):


            #print("Exp ctr è:",exp_ctr(self.t+1,advertisers))

            p_distr = [self.list_n_pulls[i][self.t-tau:].count(1)/tau for i in range(len(self.list_n_pulls))]

            #print("distr prob è:",p_distr)



            numberList = [i for i in range(self.n_arms)]

            #print("number o flist: ",numberList)

            self.last_pull = random.choices(numberList, weights=p_distr)[0]

        else: 

            #print("NORMALE")

            ucb1 = [self.avg_reward[a]+self.max_reward[a] *
                    np.sqrt(2*np.log(self.t)/self.n_pulls[a]) for a in range(self.n_arms)]
            #print(ucb1)
            self.last_pull = np.argmax(ucb1)

        new_a = self.arms[self.last_pull]
        self.n_pulls[self.last_pull] += 1

        self.list_n_pulls[self.last_pull].append(1)
        for pull in range(len(self.list_n_pulls)):      #ho aggiunto questo for per aggiungere uno 0 agli altri vettori se non sono stati pullati, mi serve per conteggiare il giusto numero di volte in cui un arm è stato tirato le ultime Tau volte
                if pull != self.last_pull:
                    self.list_n_pulls[pull].append(0)

        #print("Il numero di pulls dell arm",self.last_pull,"è: ",self.n_pulls[self.last_pull])
        self.a_hist.append(new_a)
        return new_a


    def update(self, C):  
        #print("La rewards sono:",self.rewards[self.last_pull],"con aggiunta",C)                               #C è la reward ottenuta dal mab nell'ultimo step.
        self.rewards[self.last_pull].append(C)
        if self.n_pulls[self.last_pull] == 1:
            self.avg_reward[self.last_pull] = C
            #print("primo",self.avg_reward[self.last_pull])
        else:
            self.avg_reward[self.last_pull] = (
                self.avg_reward[self.last_pull]*(self.n_pulls[self.last_pull]-1)+C)/(self.n_pulls[self.last_pull])
            #print("non primo",self.avg_reward[self.last_pull])

        self.t += 1

class UCB1_CTR_V2():
    def __init__(self, n_arms, max_reward):
        self.n_arms = n_arms
        self.max_reward = max_reward                                    #maxreward è un vettore che indica la differenza per ogni arm tra price e bid.
        self.arms = np.arange(self.n_arms)
        self.t = 0
        self.a_hist = []
        self.last_pull = None
        self.reset()

    def reset(self):
        self.t = 0
        self.last_pull = None
        self.avg_reward = np.array([np.inf for _ in range(self.n_arms)])
        self.rewards = [[] for _ in range(self.n_arms)]
        self.n_pulls = np.zeros(self.n_arms)
        self.list_n_pulls = [[] for _ in range(self.n_arms)]

        return self

    def pull_arm(self,CTR,advertisers):

        tau = 5000
        
        #print("Il time è: ",self.t)
        #print("CTR è :",CTR)


        if self.t > tau and CTR < exp_ctr(self.t+1,advertisers):


            #print("Exp ctr è:",exp_ctr(self.t+1,advertisers))

            p_distr = [self.list_n_pulls[i][self.t-tau:].count(1)/tau for i in range(len(self.list_n_pulls))]

            #print("distr prob è:",p_distr)



            numberList = [i for i in range(self.n_arms)]

            #print("number o flist: ",numberList)
            ucb1 = [self.avg_reward[a]+self.max_reward[a] *
                    np.sqrt(2*np.log(self.t)/self.n_pulls[a]) for a in range(self.n_arms)]

            self.last_pull = random.choices(numberList, weights=np.multiply(p_distr,ucb1))[0]

        else: 

            #print("NORMALE")

            ucb1 = [self.avg_reward[a]+self.max_reward[a] *
                    np.sqrt(2*np.log(self.t)/self.n_pulls[a]) for a in range(self.n_arms)]
            #print(ucb1)
            self.last_pull = np.argmax(ucb1)

        new_a = self.arms[self.last_pull]
        self.n_pulls[self.last_pull] += 1

        self.list_n_pulls[self.last_pull].append(1)
        for pull in range(len(self.list_n_pulls)):      #ho aggiunto questo for per aggiungere uno 0 agli altri vettori se non sono stati pullati, mi serve per conteggiare il giusto numero di volte in cui un arm è stato tirato le ultime Tau volte
                if pull != self.last_pull:
                    self.list_n_pulls[pull].append(0)

        #print("Il numero di pulls dell arm",self.last_pull,"è: ",self.n_pulls[self.last_pull])
        self.a_hist.append(new_a)
        return new_a


    def update(self, C):  
        #print("La rewards sono:",self.rewards[self.last_pull],"con aggiunta",C)                               #C è la reward ottenuta dal mab nell'ultimo step.
        self.rewards[self.last_pull].append(C)
        if self.n_pulls[self.last_pull] == 1:
            self.avg_reward[self.last_pull] = C
            #print("primo",self.avg_reward[self.last_pull])
        else:
            self.avg_reward[self.last_pull] = (
                self.avg_reward[self.last_pull]*(self.n_pulls[self.last_pull]-1)+C)/(self.n_pulls[self.last_pull])
            #print("non primo",self.avg_reward[self.last_pull])

        self.t += 1




