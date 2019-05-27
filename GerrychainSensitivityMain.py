
import os

os.chdir('/home/lorenzonajt/Documents/GerrychainSensitivity/PA_VTD')

from gerrychain import Graph, GeographicPartition, Partition, Election
from gerrychain.updaters import Tally, cut_edges

import numpy as np
import random
import copy

from gerrychain import MarkovChain
from gerrychain.constraints import single_flip_contiguous
from gerrychain.proposals import propose_random_flip
from gerrychain.accept import always_accept
from gerrychain.metrics import polsby_popper
from gerrychain import constraints

import matplotlib.pyplot as plt


import pandas


graph = Graph.from_file("./PA_VTD.shp")

election = Election("SEN12", {"Dem": "USS12D", "Rep": "USS12R"})

starting_partition = GeographicPartition(
    graph,
    assignment="2011_PLA_1",
    updaters={
        "polsby_popper" : polsby_popper,
        "cut_edges": cut_edges,
        "population": Tally("TOT_POP", alias="population"),
        "SEN12": election
    }
)


initial_partition = Partition(
    graph,
    assignment="2011_PLA_1",
    updaters={
        "cut_edges": cut_edges,
        "population": Tally("TOT_POP", alias="population"),
        "SEN12": election
    }
)


def analyze_dem_seats():

    d_percents = [sorted(partition["SEN12"].percents("Dem")) for partition in chain]
    data = pandas.DataFrame(d_percents)

    ax = data.boxplot(positions=range(len(data.columns)))
    data.iloc[0].plot(style="ro", ax=ax)

    plt.show()

def deviation(values):
    ideal = np.mean(values)
    deviations = [ np.abs(x - ideal)/ideal for x in values]
    return np.max(deviations)


def pop_MCMC(partition):
    temperature = 1000
    bound = 1
    if partition.parent is not None:
        parent_score = np.var(list(partition.parent["population"].values()))
        current_score = np.var(list(partition["population"].values()))
        if parent_score > current_score:
            bound = 1
        else:
            bound = (parent_score / current_score)**temperature
            #print('bound is:', bound)
    return random.random() < bound

def polsby_MCMC(partition):
    temperature = 10000
    bound = 1
    if partition.parent is not None:
        parent_score =  1 - np.mean(list(partition.parent["polsby_popper"].values()))
        current_score = 1 - np.mean(list(partition["polsby_popper"].values()))
        if parent_score > current_score:
            bound = 1
        else:
            bound = (parent_score / current_score)**(temperature)
            #print('bound is:', bound)
    return random.random() < bound


def popandpolsby_MCMC(partition):
    return polsby_MCMC(partition) and pop_MCMC(partition)

'''
def run_with_hard_constraints(starting_partition):

    pop_constraint = constraints.within_percent_of_ideal_population(initial_partition, 0.02)
    polsby_constraint = constraints.UpperBound(polsby_popper, 4)
    polsby_2 = constraints.L1_reciprocal_polsby_popper()

    chain = MarkovChain(
        proposal=propose_random_flip,
        constraints=[pop_constraint, single_flip_contiguous, polsby_2],
        accept=always_accept,
        initial_state=starting_partition,
        total_steps=10000
    )

    for part in chain:
        print(np.var(list(part["population"].values())), np.mean(list(part["polsby_popper"].values())))

    return [part]

'''

def run_with_MCMC_constraints(starting_partition):
    #pop_MCMC, polsby_MCMC

    #This is a hack : because of somethign in continuiguity check
    # starting_partition.parent = starting_partition
    #End hack

    chain = MarkovChain(
        proposal=propose_random_flip,
        constraints=[single_flip_contiguous],
        accept=popandpolsby_MCMC,
        initial_state=starting_partition,
        total_steps=100
    )

    for part in chain:
        pass
        #print(deviation(list(part["population"].values())))
    print(part.parent)
    print(part.parent is None)

    '''
    values = []
    for part in chain:
        values.append ( [deviation(list(part["population"].values())), np.mean(list(part["polsby_popper"].values()))])

    print('started', values[0])
    print('ended', values[-1])
    '''

    return part

def run_without_constraints(starting_partition):

    chain = MarkovChain(
        proposal=propose_random_flip,
        constraints=[single_flip_contiguous],
        accept=always_accept,
        initial_state=starting_partition,
        total_steps=100
    )
    for part in chain:
        pass
        #print(deviation(list(part["population"].values())))
    #print(part.parent)

    '''
    values = []
    for part in chain:
        values.append ( [deviation(list(part["population"].values())), np.mean(list(part["polsby_popper"].values()))])

    print('started', values[0])
    print('ended', values[-1])
    '''


    return part

def annealing(starting_partition):

    stages = 5

    annealed_partitions = []

    for k in range(stages):
        starting_partition = run_without_constraints(starting_partition)
        annealed_partition = run_with_MCMC_constraints(starting_partition)
        print(  [deviation(list(annealed_partition["population"].values())), np.mean(list(annealed_partition["polsby_popper"].values()))] )
        annealed_partitions.append(annealed_partition)
        starting_partition = annealed_partition

    for part in annealed_partitions:
        print(  [deviation(list(annealed_partition["population"].values())), np.mean(list(annealed_partition["polsby_popper"].values()))] )

    return True