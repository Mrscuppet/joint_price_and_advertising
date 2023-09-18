import numpy as np


class Enviroment:
    def __init__(self,k,allocation_mec,user_generator):
        self.k = k                                                       #lunghezza di ogni batch ( numero di ad che compaiono in un singolo batch).
        self.allocation_mec = allocation_mec                             #meccanismo di allocazione usato per ottenere lambda ( vettore che indica gli slot degli ad da quali advertiser sono occupati), restituisce una lista di namedtuple: Competitor/Agent(id, bid, price).
        self.user_generator = user_generator                             #generatore che genera lo user.

    def step(self,comp_prices_bids,agent_price_bid):
        u = self.user_generator()   #LUNGHEZZA TOLTA PER ORA PER TEST                             #la lunghezza di user_gen deve essere quanto quella dei competitors +1,restiuisce vettore user con primo elemento indicante quanti batch vede lo user, e gli altri il "quality score" dello user per ognic competitor.
        print("Generated user: ",u,"\n")                                
        positions = self.allocation_mec(comp_prices_bids,agent_price_bid)
        visit = positions[:u[0]*self.k]                                  #dalla lista degli ad ordinati secondo il meccanismo di allocazione ci si prende solo quelli visti dallo user

        print("Visit by user: ",visit,"\n")

        visit_prices = []
        for i in range(len(visit)):
            visit_prices.append(visit[i].price)                          #vettore prezzi visti dallo user, in base ovviamente agli item visti


        print("Visit prices by user: ",visit_prices,"\n")
        """
        
        visit_ref_prices = []                                            #calcolo dei reference_prices cioè i prezzi pesati in base al "quality score" che un utente associa ai vari competitor
        for i in range(len(visit_prices)):
            if type(visit[i]) == type(agent_price_bid):                  #se il prezzo visto è dell'agente si calcola il reference price con il prezzo*l'ultimo elemento di user che è quello che si è scelto di usare per la quality score dell'agent
                visit_ref_prices.append(visit_prices[i]*u[-1])
            else:
                visit_ref_prices.append(visit_prices[i]*u[visit[i].id+1])#vettore dei reference prices calcolato come prezzi visti dallo user * coefficiente "d'attrazione" dello user u per quel competitor, il + 1 è dovuto al fatto che il primo elem di usero indica in num di batches visitati
    """
        visit_ref_prices = []                                            #calcolo dei reference_prices cioè i prezzi pesati in base al "quality score" che un utente associa ai vari competitor
        for i in range(len(visit_prices)):
                visit_ref_prices.append(visit_prices[i]*u[visit[i].id+1])
    
        print("References prices: ",visit_ref_prices,"\n")
        
        index_of_min = np.where(visit_ref_prices == np.amin(visit_ref_prices))#index_of_min[0] è un array che contiene l'indice del minimo ref price ( o dei minimi se sono più di uno )
        
        reward_and_id = []                                               #vettore che conterrà la reward dell'agente e l'id dell'agent scelto dal MAB
        """
        if len(index_of_min[0]) > 1:

            print("Multiplo vincitore,scelta random: \n")                #in caso ci sono due referenc prices uguali lo user scegliere un item randomico

            num_ran = np.random.choice(index_of_min[0], size=1)          

            print("Type dello scelto: ",type(visit[num_ran]),"\n")
            if type(visit[num_ran]) == type(comp_prices_bids[0]):        #Se l'ad scelto è di un competiro allora la reward sarà 0, e ci salviamo l'id dell'agente che ci serve per utilizzo del mab successivamente ( in particolare per EXP3 )
                print("Competitor è stato scelto")
                reward_and_id.append(0)
                reward_and_id.append(agent_price_bid.id)
                print(agent_price_bid.id)
                return reward_and_id
            else:
                print("Agent è stato scelto")                            #Se l'ad scelto è invece dell'agente si esegue questo else, NOTA: in questo caso è la stessa cosa usare reward_and_id.append(visit[index_of_min[0][0]].id) o reward_and_id.append(agent_price_bid.id)
                diff = visit[num_ran].price - visit[num_ran].bid
                reward_and_id.append(diff)
                reward_and_id.append(visit[index_of_min[0][0]].id)
                print(visit[index_of_min[0][0]].id)
                return reward_and_id

        else:                                                           #Else che gestisce il caso non ci sia un pari merito, simile poi tutto alla prima parte dell if.

            print("Singolo vincitore, non random: \n")
            print("Type dello scelto: ",type(visit[index_of_min[0][0]]),"\n")
            if type(visit[index_of_min[0][0]]) == type(comp_prices_bids[0]):
                print("Competitor è stato scelto")
                print("La reward dell'Agent è 0\n")
                reward_and_id.append(0)
                reward_and_id.append(agent_price_bid.id)
                print("L'id dell'agent è: ",agent_price_bid.id,"\n")
                return reward_and_id
            else:
                print("Agent è stato scelto")
                diff = visit[index_of_min[0][0]].price - visit[index_of_min[0][0]].bid
                print("La reward dell'agent è ovviamente :",diff,"\n")
                reward_and_id.append(diff)
                reward_and_id.append(visit[index_of_min[0][0]].id)
                print("L'id dell'agent è: ",visit[index_of_min[0][0]].id,"\n")
                return reward_and_id"""
    
        if len(index_of_min[0]) > 1:

                print("Multiplo vincitore,scelta random: \n")                #in caso ci sono due referenc prices uguali lo user scegliere un item randomico

                num_ran = np.random.choice(index_of_min[0], size=1)          

                print("Type dello scelto: ",type(visit[num_ran]),"\n")
                print("Agent è stato scelto")                            #Se l'ad scelto è invece dell'agente si esegue questo else, NOTA: in questo caso è la stessa cosa usare reward_and_id.append(visit[index_of_min[0][0]].id) o reward_and_id.append(agent_price_bid.id)
                diff = visit[num_ran].price - visit[num_ran].bid
                reward_and_id.append(diff)
                reward_and_id.append(visit[index_of_min[0][0]].id)
                print("L'id dell'agent è: ",visit[index_of_min[0][0]].id,"\n")
                return reward_and_id

        else:                                                           #Else che gestisce il caso non ci sia un pari merito, simile poi tutto alla prima parte dell if.

                print("Singolo vincitore, non random: \n")
                print("Type dello scelto: ",type(visit[index_of_min[0][0]]),"\n")
                print("Agent è stato scelto")
                diff = visit[index_of_min[0][0]].price - visit[index_of_min[0][0]].bid
                print("La reward dell'agent è ovviamente :",diff,"\n")
                reward_and_id.append(diff)
                reward_and_id.append(visit[index_of_min[0][0]].id)
                print("L'id dell'agent è: ",visit[index_of_min[0][0]].id,"\n")
                return reward_and_id


