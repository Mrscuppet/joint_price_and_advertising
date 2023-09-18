import random
from collections import namedtuple
import copy
import numpy as np

#Funzione per allocation mechanism

def allocation_mec(comps_prices_bids,age_price_bid,CTR):   

        ncomp = namedtuple('Competitor', ['id', 'price','bid','CTR'])
        nagen = namedtuple('Agent', ['id', 'price','bid','CTR'])

        all = copy.deepcopy(comps_prices_bids)
        all.append(age_price_bid)
        all   
        nshops = []
        for i in range(len(all)-1):
                nshops.append(ncomp(i,all[i].price,all[i].bid,CTR[i]))
        nshops.append(nagen(all[-1].id,all[-1].price,all[-1].bid,CTR[-1]))
        position = sorted(nshops, key=lambda x: x[3]*x[2], reverse=True)                                   
        #print("Products displayed by Google: ",position,"\n")

        return position
    
#Uguale all'altro all mec ma senza print per le funzioni per il calcolo della regret

def allocation_mec_stability(comps_prices_bids,CTR):   

        ncomp = namedtuple('Competitor', ['id', 'price','bid','CTR'])

    
        nshops = []
        for i in range(len(comps_prices_bids)-1):
                nshops.append(ncomp(i,comps_prices_bids[i].price,comps_prices_bids[i].bid,CTR[i]))
        position = sorted(nshops, key=lambda x: x[3]*x[2], reverse=True)                                   
        #print("Products displayed by Google: ",position,"\n")

        return position

def allocation_mec_regret(comps_prices_bids,age_price_bid,CTR):   

        ncomp = namedtuple('Competitor', ['id', 'price','bid','CTR'])
        nagen = namedtuple('Agent', ['id', 'price','bid','CTR'])

        all = copy.deepcopy(comps_prices_bids)
        all.append(age_price_bid)
        all   
        nshops = []
        for i in range(len(all)-1):
                nshops.append(ncomp(i,all[i].price,all[i].bid,CTR[i]))
        nshops.append(nagen(all[-1].id,all[-1].price,all[-1].bid,CTR[-1]))
        position = sorted(nshops, key=lambda x: x[3]*x[2], reverse=True)                                   
        #print("Products displayed by Google: ",position,"\n")

        return position


def CTR_update_stability(t,reward,CTR):
    n = [1 for _ in range(len(CTR))]             #ho aggiunto questo if poichè gli id dell agent sono diversi per EXP3, e quindi mi sballava il calcolo del CTR
    for i in range(len(CTR)):
        if reward[2] == i:
            n[i] = 1
        else:
            n[i] = 0
    for i in range(len(CTR)):
        CTR[i] = CTR[i] + (n[i] - CTR[i])/t
    #print("Stampa dei CTR: ",CTR,"\n")
    return CTR
    


#Aggiornamento CTR

def CTR_update(t,reward,CTR):
    n = [1 for _ in range(len(CTR))]
    if reward[2] >= len(CTR)-1:
        reward[2] = len(CTR)-1               #ho aggiunto questo if poichè gli id dell agent sono diversi per EXP3, e quindi mi sballava il calcolo del CTR
    for i in range(len(CTR)):
        if reward[2] == i:
            n[i] = 1
        else:
            n[i] = 0
    for i in range(len(CTR)):
        CTR[i] = CTR[i] + (n[i] - CTR[i])/t
    #print("Stampa dei CTR: ",CTR,"\n")
    return CTR

#Come l'altro CTR ma senza print per la regret

def CTR_update_regret(t,reward,CTR):
    n = [1 for _ in range(len(CTR))]
    if reward[2] >= len(CTR)-1:
        reward[2] = len(CTR)-1               
    for i in range(len(CTR)):
        if reward[2] == i:
            n[i] = 1
        else:
            n[i] = 0
    for i in range(len(CTR)):
        CTR[i] = CTR[i] + (n[i] - CTR[i])/t
    #print("Stampa dei CTR: ",CTR,"\n")
    return CTR

#Funzione che genera lo user, il primo elemento sono il numero di batches che vede. Gli altri indicano il coefficiente di "affidabilità" del competitor

def user_gernator_old(mbs ,var):  #max_batch_seen, #variance

    u = []
    u.append(random.randint(1,mbs))       
    u.append(np.random.normal(0.948,var))
    u.append(np.random.normal(0.96,var))
    u.append(np.random.normal(1,var))
    u.append(np.random.normal(1,var))
    u.append(np.random.normal(1,var))
    u.append(np.random.normal(0.9,var))

    return u

def user_gernator(nobi,nobs,mean,var):  #number_of_batches_inferior,#number_of_batches_superior,#mean, #variance

    u = []
    u.append(int(np.random.uniform(nobi,nobs)))
    #u.append(5) #per simulare che li vede tutti
    for i in range(len(mean)):    
        u.append(np.random.normal(mean[i],var))
     
    return u

    
#Funzione generatrice dell'agent price e bids.

def agent_prices_and_bids_old(num):
    comp = namedtuple('Agent', ['id', 'price','bid'])
    agent = [comp(i,(random.randint(1,20)),round(random.uniform(0.5,5), 2)) for i in range(num)]

    return agent

#Funzione generatrice di price e bid per l'agent con i prices normalizzati nello stesso intervallo delle bid

def agent_prices_and_bids(num):
    comp = namedtuple('Agent', ['id', 'price','bid'])
    agent = [comp(i,random.randint(1,20),round(random.uniform(0.5,5), 2)) for i in range(num)]      #le bid vengono generate tra 0,5 e 5 e i prezzi variano tra 1 e 20 ma sono normalizzati nello stesso intervallod delle bid in modo da poter fare operazioni sullo stesso "ranage"
    norm_agent = norm_prices(agent)

    return norm_agent



#Funzione generatrice dei competitors price e bids.
#Num corrisponde al numero dei competitors poichè considero una sola coppia bid e price per competitor

def competitors_prices_and_bids_old(num):
    comp = namedtuple('Competitor', ['id', 'price','bid'])
    competitors = [comp(i,random.randint(1,20),round(random.uniform(0.5,5), 2)) for i in range(num)]

    return competitors


#Funzione generatrice di price e bid con i prices normalizzati nello stesso intervallo delle bid

def competitors_prices_and_bids(num):
    comp = namedtuple('Competitor', ['id', 'price','bid'])
    competitors = [comp(i,random.randint(1,20),round(random.uniform(0.5,5), 2)) for i in range(num)] #le bid vengono generate tra 0,5 e 5 e i prezzi variano tra 1 e 20 ma sono normalizzati nello stesso intervallod delle bid in modo da poter fare operazioni sullo stesso "ranage"
    norm_competitors = norm_prices(competitors)

    return norm_competitors


#Funzione per allocation mechanism
def allocation_mec_old(comps_prices_bids,age_price_bid):                                                #gli attributi sono la lista di namedtuple dei competitor e il namedtuple dell'agent scelto dal mab
        appoggio = copy.deepcopy(comps_prices_bids)
        appoggio.append(age_price_bid)
        position = sorted(appoggio, key=lambda x: int(x[1]-x[2]))                                   #restiuisce il vettore di namedtuple in base al criterio x considerato ( per ora price - bid)
        print("Prices displayed by Google: ",position,"\n")

        return position
        

#Funziona per generare lo user
#num indica il numero di competitors

def user_gernator_old(num):
    u = []
    u.append(random.randint(1,5))                                                                   #numero di batch che l'utente vuol vedere
    for _ in range(num):
        u.append(random.random())                                                                   #genera il "quality score" di ogni competitor per lo user, l'ultimo è quello che si considera che fa riferimento all'agent

    return u

#Funzione per normalizzare i prezzi 

def norm_prices(competitors):
    prices = [competitors[i].price for i in range(len(competitors))]
    maxx = max(prices)
    minn = min(prices)
    new_vals = []
    for i in range(len(competitors)):
        new_vals.append(normalization_range(competitors[i].price,maxx,minn,0.5,5))                  #0,5 e 5 è l'intervallo che si è scelto in cui normalizzare i valori
        competitors[i] = competitors[i]._replace(price = round((new_vals[i]),2))                    
    
    return competitors 


#Funzione che normalizza in un determinato intervallo, dove a_ra e b_ra sono gli estremi dell'intervallo

def normalization_range(val,max,min,a_ra,b_ra):
    normalized = a_ra + ((val-min)*(b_ra - a_ra))/(max-min)
    return normalized

