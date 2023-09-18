import numpy as np
import utility as ut
from collections import namedtuple
import copy


class Environment:
    def __init__(self,k,t,allocation_mec,user_generator,CTR):
        self.k = k                                                       #lunghezza di ogni batch (numero di ad che compaiono in un singolo batch).
        self.allocation_mec = allocation_mec                             #meccanismo di allocazione usato per ottenere lambda ( vettore che indica gli slot degli ad da quali advertiser sono occupati), restituisce una lista di namedtuple: Competitor/Agent(id, bid, price).
        self.user_generator = user_generator                             #generatore che genera lo user.
        self.t = t                                                       #time istat considerato che influisce sul calcolo del CTR, è 20, ma il primo voglio ceh sia 22 per il ctr, quindi alcuni campi avrà +1
        self.CTR = CTR                                                   #vettore che indica CTR di ogni advertiser

    def step(self,comp_prices_bids,agent_price_bid):

        prints = [] #vettore per salvarm le print 

        #print("\n\n|| ITERAZIONE NUMERO:",self.t - 19,"\n")
        prints.append("\n\n|| ITERAZIONE NUMERO:")
        prints.append(str(self.t - 19))
        prints.append("\n")

        nozeromean = [0.9,0.9,0.9,0.9,1,1,1,1,1]
        
        

        #u = self.user_generator(1,3,np.ones(9),0.2)                                       #la lunghezza di user_gen deve essere quanto quella dei competitors +1,restiuisce vettore user con primo elemento indicante quanti batch vede lo user, e gli altri il "quality score" dello user per ogni competitor.
        u = self.user_generator(1,2,np.ones(10),0.2)                                       
        prints.append("Generated user: ")
        prints.append(str(u))
        prints.append("\n")
                  
        positions = self.allocation_mec(comp_prices_bids,agent_price_bid,self.CTR)
        #print("Products displayed by Google: ",position,"\n")
        prints.append("Products displayed by Google: ")
        prints.append(str(positions))
        prints.append("\n")
                  
        visit = positions[:u[0]*self.k]                                  #dalla lista degli ad ordinati secondo il meccanismo di allocazione ci si prende solo quelli visti dallo user


        comp = namedtuple('Agent', ['id', 'price','bid'])
        comp1 = namedtuple('Competitor', ['id', 'price','bid'])

        Shop0 = comp1(0,8,2.5)
        ag0 = comp(5,5.2,2.2)

        type(Shop0).__name__
        type(ag0).__name__


        #Pezzo di codice per salvarsi la posizione dell'agent negli slot visualizzati dall'utente
        check = 0
        for elem in range(len(visit)):
            if (type(visit[elem]).__name__ == type(ag0).__name__):
                #print("Lo slot assegnato all'agent è il numero:",elem,"\n")
                prints.append("Lo slot assegnato all'agent è il numero:")
                prints.append(str(elem))
                prints.append("\n")
                pos = elem
                check = 1
        if check == 0:
            #print("Agent non presente, assegnato slot numero",-1,"\n")
            prints.append("Agent non presente, assegnato slot numero,-1,\n")

            pos = -1

        """
        bool = 0
        user_attr = copy.deepcopy(u)
        user_attr.pop(0)
        if (user_attr.index(max(user_attr)) == user_attr.index(user_attr[-1])):
            print("L'agent ha l'attractiveness coefficient più alto")
            bool = 1"""


        #print("Products seen by the user: ",visit,"\n")
        prints.append("Products seen by the user: ")
        prints.append(str(visit))
        prints.append("\n")

        visit_prices = []
        for i in range(len(visit)):
            visit_prices.append(visit[i].price)                          #vettore prezzi visti dallo user, in base ovviamente agli item visti


        #print("Prices seen by user: ",visit_prices,"\n")

        prints.append("Prices seen by user: ")
        prints.append(str(visit_prices))
        prints.append("\n")
        
        visit_ref_prices = []                                            #calcolo dei reference_prices cioè i prezzi pesati in base al "quality score" che un utente associa ai vari competitor
        for i in range(len(visit_prices)):
            if type(visit[i]).__name__ == type(agent_price_bid).__name__:                  #se il prezzo visto è dell'agente si calcola il reference price con il prezzo*l'ultimo elemento di user che è quello che si è scelto di usare per la quality score dell'agent
                visit_ref_prices.append(visit_prices[i]*u[-1])
            else:
                visit_ref_prices.append(visit_prices[i]*u[visit[i].id+1])#vettore dei reference prices calcolato come prezzi visti dallo user * coefficiente "d'attrazione" dello user u per quel competitor, il + 1 è dovuto al fatto che il primo elem di usero indica in num di batches visitati
   
   
        #print("References prices perceived by the user: ",visit_ref_prices,"\n")
        prints.append("References prices perceived by the user: ")
        prints.append(str(visit_ref_prices))
        prints.append("\n")
        
        
        index_of_min = np.where(visit_ref_prices == np.amin(visit_ref_prices))#index_of_min[0] è un array che contiene l'indice del minimo ref price ( o dei minimi se sono più di uno )
        
        reward_and_id = []    
        
        #print(agent_price_bid.id)                                           #vettore che conterrà la reward dell'agente ( 0 se non è stato scelto, altrimenti bid - price ) e l'id del ad scelto dallo user
        #print(visit[index_of_min[0][0]].id)
        if len(index_of_min[0]) > 1:

            #print("Multiplo vincitore,scelta random:\n")                #in caso ci sono due referenc prices uguali lo user scegliere un item randomico
            prints.append("Multiplo vincitore,scelta random:\n")

            num_ran = np.random.choice(index_of_min[0], size=1)   

            print(num_ran)       

            #print("Type dello scelto: ",type(visit[num_ran]),"\n")
            if type(visit[num_ran[0]]) == type(comp_prices_bids[0]):        #Se l'ad scelto è di un competiro allora la reward sarà 0, e ci salviamo l'id dell'agente che ci serve per utilizzo del mab successivamente ( in particolare per EXP3 )
                #print("Competitor è stato scelto\n")
                #print("La reward dell'Agent è 0\n")
                prints.append("Competitor è stato scelto\nLa reward dell'Agent è 0\n")
                reward_and_id.append(0)
                reward_and_id.append(agent_price_bid.id)
                #print("CIAOOO QUESTO è L ID DELLO SCELTO:",visit[index_of_min[0][0]].id)
                reward_and_id.append(visit[index_of_min[0][0]].id)
                reward_and_id.append(u)
                reward_and_id.append(pos)
                reward_and_id.append(0)

                bool = 0
                user_attr = copy.deepcopy(u)
                user_attr.pop(0)
                if (user_attr.index(max(user_attr)) == user_attr.index(user_attr[-1])):
                    #print("L'agent ha l'attractiveness coefficient più alto, ma non è stato scelto\n")
                    prints.append("L'agent ha l'attractiveness coefficient più alto, ma non è stato scelto\n")
                    bool = 1
                #print("L'id del competitor scelto è: ",visit[index_of_min[0][0]].id,"\n")
                prints.append("L'id del competitor scelto è: ")
                prints.append(str(visit[index_of_min[0][0]].id))
                prints.append("\n")
                self.CTR = ut.CTR_update(self.t+1,reward_and_id,self.CTR)
                #print("Stampa dei CTR: ",CTR,"\n")
                prints.append("Stampa dei CTR: ")
                prints.append(str(self.CTR))
                prints.append("\n")

                self.t = self.t + 1
                reward_and_id.append(prints)

                return reward_and_id
            else:
                #print("Agent è stato scelto\n")                            #Se l'ad scelto è invece dell'agente si esegue questo else, NOTA: in questo caso è la stessa cosa usare reward_and_id.append(visit[index_of_min[0][0]].id) o reward_and_id.append(agent_price_bid.id)
                diff = visit[num_ran[0]].price - visit[num_ran[0]].bid
                #print("La reward dell'agent è:",diff,"\n")
                prints.append("Agent è stato scelto\n")
                prints.append("La reward dell'agent è:")
                prints.append(str(diff))
                prints.append("\n")
                reward_and_id.append(diff)
                reward_and_id.append(agent_price_bid.id)
                reward_and_id.append(visit[index_of_min[0][0]].id)
                reward_and_id.append(u)
                reward_and_id.append(pos)
                bool = 0
                user_attr = copy.deepcopy(u)
                user_attr.pop(0)
                if (user_attr.index(max(user_attr)) == user_attr.index(user_attr[-1])):
                    #print("L'agent ha l'attractiveness coefficient più alto, ed è stato anche scelto\n")
                    prints.append("L'agent ha l'attractiveness coefficient più alto, ed è stato anche scelto\n")
                    bool = 1
                if bool == 1:
                    reward_and_id.append(bool)   #è stato scelto l'agent che ha anche l'attraxtiveness coefficient più alto
                else:
                    reward_and_id.append(0)
                #print("L'id dell'agent è: ",agent_price_bid.id,"\n")
                prints.append("L'id dell'agent è: ")
                prints.append(agent_price_bid.id)
                prints.append("\n")
                self.CTR = ut.CTR_update(self.t+1,reward_and_id,self.CTR)
                prints.append("Stampa dei CTR: ")
                prints.append(str(self.CTR))
                prints.append("\n")
                self.t = self.t + 1
                reward_and_id.append(prints)

                return reward_and_id

        else:                                                           #Else che gestisce il caso non ci sia un pari merito, simile poi tutto alla prima parte dell if.

            #print("Singolo vincitore:\n")
            #print("Type dello scelto: ",type(visit[index_of_min[0][0]]),"\n")

            if type(visit[index_of_min[0][0]]).__name__ == type(comp_prices_bids[0]).__name__:  #check che serve a capire se il vincitore è un Competitor o è l'Agent

                #print("Competitor è stato scelto\n")
                #print("La reward dell'Agent è 0\n")
                prints.append("Singolo vincitore:\n")
                prints.append("Competitor è stato scelto\n")
                prints.append("La reward dell'Agent è 0\n")
                #print("STO NELL IF",visit[index_of_min[0][0]].id)


                reward_and_id.append(0)
                reward_and_id.append(agent_price_bid.id)
                #print("CIAOOO QUESTO è L ID DELLO SCELTO:",visit[index_of_min[0][0]].id)
                reward_and_id.append(visit[index_of_min[0][0]].id)
                reward_and_id.append(u)
                reward_and_id.append(pos)
                reward_and_id.append(0)
                reward_and_id.append(prints)
                bool = 0
                user_attr = copy.deepcopy(u)
                user_attr.pop(0)
                if (user_attr.index(max(user_attr)) == user_attr.index(user_attr[-1])):
                    #print("L'agent ha l'attractiveness coefficient più alto, ma non è stato scelto\n")
                    prints.append("L'agent ha l'attractiveness coefficient più alto, ma non è stato scelto\n")
                    bool = 1
                #print("L'id del competitor scelto è: ",visit[index_of_min[0][0]].id,"\n")
                prints.append("L'id del competitor scelto è: ")
                prints.append(str(visit[index_of_min[0][0]].id))
                prints.append("\n")
                self.CTR = ut.CTR_update(self.t+1,reward_and_id,self.CTR)
                prints.append("Stampa dei CTR: ")
                prints.append(str(self.CTR))
                prints.append("\n")
                self.t = self.t + 1

                reward_and_id.append(prints)

                return reward_and_id

            else:

                #print("Agent è stato scelto\n")
                diff = visit[index_of_min[0][0]].price - visit[index_of_min[0][0]].bid
                ##print("La reward dell'agent è:",diff,"\n")
                prints.append("Agent è stato scelto\n")
                prints.append("La reward dell'agent è:")
                prints.append(str(diff))
                prints.append("\n")
                #print("STO NELL ELSE",visit[index_of_min[0][0]].id)

                reward_and_id.append(diff)
                reward_and_id.append(agent_price_bid.id)
                #print(visit[index_of_min[0][0]].id)
                reward_and_id.append(visit[index_of_min[0][0]].id)
                reward_and_id.append(u)
                reward_and_id.append(pos)
                bool = 0
                user_attr = copy.deepcopy(u)
                user_attr.pop(0)
                if (user_attr.index(max(user_attr)) == user_attr.index(user_attr[-1])):
                    #print("L'agent ha l'attractiveness coefficient più alto, ed è stato anche scelto\n")
                    prints.append("L'agent ha l'attractiveness coefficient più alto, ed è stato anche scelto\n")
                    bool = 1
                if bool == 1:
                    reward_and_id.append(bool)   #è stato scelto l'agent che ha anche l'attraxtiveness coefficient più alto
                else:
                    reward_and_id.append(0)
                #print("L'id dell'agent è: ",agent_price_bid.id,"\n")
                prints.append("L'id dell'agent è: ")
                prints.append(str(agent_price_bid.id))
                prints.append("\n")
                self.CTR = ut.CTR_update(self.t+1,reward_and_id,self.CTR)
                prints.append("Stampa dei CTR: ")
                prints.append(str(self.CTR))
                prints.append("\n")
                self.t = self.t + 1
                
                reward_and_id.append(prints)

                return reward_and_id
    



