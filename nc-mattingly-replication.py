
# coding: utf-8

# In[192]:


# Imports

import os

from gerrychain import Graph, GeographicPartition, Partition, Election
from gerrychain.updaters import Tally, cut_edges
import geopandas as gpd
import numpy as np
from gerrychain.random import random
import copy

from gerrychain import MarkovChain
from gerrychain.constraints import single_flip_contiguous, no_vanishing_districts
from gerrychain.proposals import propose_random_flip
from gerrychain.accept import always_accept
from gerrychain.metrics import polsby_popper
from gerrychain import constraints

from collections import defaultdict, Counter

import matplotlib.pyplot as plt

import networkx as nx

import pandas

import math

#from IPython.display import clear_output

from functools import partial


# In[217]:

count = 0
class MH_Annealer:
    def __init__(self, temp):
        self.counter = 0 # counts number of steps in the chain, increment on accept
        self.temp = temp # function which eats a count and gives the current temp
        
        
        
        
    def __call__(self, partition):
        #partition["score"] = self.score(partition)
        # if self.temp(self.counter) != 0 : prob = (len(partition.parent["cut_edges"])/len(partition["cut_edges"]) * 
        #         math.exp(-self.temp(self.counter) * (partition["score"] - partition.parent["score"])))
        # else: prob = (len(partition.parent["cut_edges"])/len(partition["cut_edges"]))
        prob = (len(partition.parent["cut_edges"])/len(partition["cut_edges"]) * 
                 math.exp(-self.temp(self.counter) * (partition["score"] - partition.parent["score"])))  
        #print((len(partition["cut_edges"])/len(partition.parent["cut_edges"]),prob)
        if prob > random.random():
            self.counter += 1
            global count
            count +=1
            return True
        else:
            return False






def mattinglyscore_2018(partition, df):
    global count
    if count < 39900: return 0
    # Based on this one https: // arxiv.org / pdf / 1801.03783.pdf
    # This is the most polished version available


    M_C = 1000


    ###Population Part

    pop_list = list(partition["population"].values())
    J_p = np.linalg.norm(( [ x/ pop_ideal - 1 for x in pop_list]))


    ###PolsbyPopperPart
    J_I = np.sum( list( partition["polsby_popper"]))

    ###County Splits
    
#     tups = {v:k for k,d in partition.assignment.parts.items() for v in d}
    
#     df["current"] = df["indx"].map(tups)
#     #print(partition.assignment.parts)
#     county_splits = defaultdict(list) #Maps to list of number of VTDS in each districts

    

#     for pair in df.groupby(["County", "current"])["indx"]:
#         county_splits[pair[0][0]].append(len(pair[1]))
    # Checking against num splits: len ( [x for x in county_splits.keys() if len(county_splits[x]) > 1] )
    #print(county_splits)
    num_2_splits = 0
    num_2_splits_W = 0
    num_greater_splits = 0
    num_greater_splits_W = 0

    county_splits = {k:[] for k in counties}
    dct = {  k:[countydict[v] for v in d] for k,d in partition.assignment.parts.items()   }
    dct = {k: Counter(v) for k,v in dct.items()}
    
    for v in dct.values():
        for k,ct in v.items():
            county_splits[k].append(ct)
            
    #print(county_splits)
    
    for county in county_splits.keys():
        if len( county_splits[county]) == 2:
            total = sum( county_splits[county])
            max_2 = min( county_splits[county])
            num_2_splits += 1
            num_2_splits_W += np.sqrt( max_2 / total )
        elif len(county_splits[county]) > 2:
            total = sum(county_splits[county])
            county_splits[county].sort()
            left_overs = total - county_splits[county][-1] - county_splits[county][-2]
            num_greater_splits += 1
            num_greater_splits_W += np.sqrt( left_overs / total)

    J_c = num_2_splits * num_2_splits_W + M_C * num_greater_splits * num_greater_splits_W
    # mattingly's "poor man's vra" is sqrt(H(.4448 - m_1)) + sqrt(H(.3620 - m_2)) where
    # m_1 and m_2 are the BPOP for the two districts with the highest BPOP
    # and H(x) = max(0,x) (the name H comes from 'hinge', bc this looks like the hinge loss in ML)
    
    
    pop_list = list(partition["population"].values())
    bpop_list = list(partition["black_pop"].values())
    
    
    bpop_props = sorted([ bpop_list[i]/pop_list[i] for i in range(len(pop_list)) ], reverse=True)
    
    
    J_m = math.sqrt( max(0,(.4448 - bpop_props[0]))   ) + math.sqrt(  max(0,.3620 - bpop_props[1])  )
    return 3000 * J_p + 2.5 * J_I + 0.4 * J_c + 800 * J_m



# In[195]:


# ## Methods

# def analyze_dem_seats():

#     d_percents = [sorted(partition["SEN12"].percents("Dem")) for partition in chain]
#     data = pandas.DataFrame(d_percents)

#     ax = data.boxplot(positions=range(len(data.columns)))
#     data.iloc[0].plot(style="ro", ax=ax)

#     plt.show()

# def deviation(values):
#     ideal = np.mean(values)
#     deviations = [ np.abs(x - ideal)/ideal for x in values]
#     return np.max(deviations)

# def MattinglyScore_2014(partition, L = 1/2, beta = 1):
#     #L is Mattingly's Lambda

#     #Based on this one: https://arxiv.org/pdf/1410.8796.pdf
#     #This was the original project, and was a prototype

#     c_pop = 1/5000
#     c_compact = 2000

#     J_pop = c_pop*np.var(list(partition["population"].values()))
#     J_compact = c_compact * np.sum( list( partition["polsby_popper"]))

#     J_L = L * J_pop + (1 - L) * J_compact


#     return np.exp(e, -1 * beta * J_L)



# def num_splits(partition, df):
#     df["current"] = df.index.map(dict(partition.assignment))
#     return sum(df.groupby("COUNTYFP10")["current"].nunique() > 1)

# def pop_MCMC(partition):
#     temperature = 1000
#     bound = 1
#     if partition.parent is not None:
#         parent_score = np.var(list(partition.parent["population"].values()))
#         current_score = np.var(list(partition["population"].values()))
#         if parent_score > current_score:
#             bound = 1
#         else:
#             bound = (parent_score / current_score)**temperature
#             #print('bound is:', bound)
#     return random.random() < bound

# def polsby_MCMC(partition):
    
#     temperature = 100000
#     bound = 1
#     if partition.parent is not None:
#         parent_score =  1 - np.mean(list(partition.parent["polsby_popper"].values()))
#         current_score = 1 - np.mean(list(partition["polsby_popper"].values()))
#         if parent_score > current_score:
#             bound = 1
#         else:
#             bound = (parent_score / current_score)**(temperature)
#             #print('bound is:', bound)
#     return random.random() < bound


# def popandpolsby_MCMC(partition):
#     return polsby_MCMC(partition) and pop_MCMC(partition)




def mattingly_temperature(counter):
    # temp is zero for the first 40,000 steps
    # increaes linearly to one over the next 60,000 accepted steps
    # fixed at one for 20,000 accepted steps, then we record the plan
    
    if counter < 40000: return 0
    elif counter > 100000: return 1
    else: return (counter-40000)/60000
        
    


# In[196]:


# setup -- SLOW

shapefile = "NC_VTD/NC_VTD.shp"

df = gpd.read_file(shapefile)



# for idx, row in df.iterrows():
#     try:
#         row.geometry.intersection(row.geometry)
#     except gpd.TopologicalError:
#         buffered = row.geometry.buffer(0)
#         buffered.intersection(buffered)
#         repaired.append(idx)
#         row.geometry = buffered

        
graph = Graph.from_geodataframe(df,ignore_errors=True)
df['indx'] = range(len(df))
  


# In[197]:


graph.add_data(df,list(df))
print(list(df))
counties = (set(list(df["County"])))
print(len(counties))
countydict = dict(graph.nodes(data='County'))
print(countydict)


# In[198]:


election = Election("SEN12", {"Dem": "EL14G_USS_", "Rep": "EL14G_US_1"})




    
    



starting_partition = GeographicPartition(
    graph,
    assignment="judge",
    updaters={
        "polsby_popper" : polsby_popper,
        "cut_edges": cut_edges,
        "population": Tally("PL10AA_TOT", alias="population"),
        "black_pop": Tally("BPOP", alias="black_pop"), #this is black pop, for BVAP, use 'BVAP' (nb: BVAP doesn't include black hispanic but BPOP does)
        "SEN12": election,
        "score" : partial(mattinglyscore_2018, df=df)
        #"County Splits": num_splits
    }
)

print(single_flip_contiguous(starting_partition))


# In[ ]:


#run with MCMC constraints

#This is a hack : because of somethign in continuiguity check
#starting_partition.parent = starting_partition
#End hack

mattingly_accept = MH_Annealer( temp = mattingly_temperature)

pop_list = list(starting_partition["population"].values())
pop_ideal = np.sum ( pop_list) / 13


chain = MarkovChain(
    proposal=propose_random_flip,
    constraints=[single_flip_contiguous, no_vanishing_districts],
    accept=mattingly_accept,
    initial_state=starting_partition,
    total_steps=120000
)
import time
tic = time.time()
step = 0
for part in chain:
    step +=1
    print(step,end='\r')
    #print(mattinglyscore_2018(part,df))
    #print("SCORE: {}  STEP: {}".format( mattinglyscore_2018(part,df),step))
    #print(mattinglyscore_2018(part,df))
    #print(deviation(list(part["population"].values())))
    #if step == 120000: print(mattinglyscore_2018(part,df))
    #clear_output(wait=True)
print(time.time() - tic)

