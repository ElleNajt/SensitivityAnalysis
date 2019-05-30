
import os

os.chdir('/home/lorenzonajt/Documents/GerrychainSensitivity/PA_VTD')

from gerrychain import Graph, GeographicPartition, Partition, Election
from gerrychain.updaters import Tally, cut_edges
import geopandas as gpd
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

def MattinglyScore_2014(partition, L = 1/2, beta = 1):
    #L is Mattingly's Lambda

    #Based on this one: https://arxiv.org/pdf/1410.8796.pdf
    #This was the original project, and was a prototype

    c_pop = 1/5000
    c_compact = 2000

    J_pop = c_pop*np.var(list(partition["population"].values()))
    J_compact = c_compact * np.sum( list( partition["polsby_popper"]))

    J_L = L * J_pop + (1 - L) * J_compact


    return np.exp(e, -1 * beta * J_L)

def MattinglyScore_2018(partition, df):
    # Based on this one https: // arxiv.org / pdf / 1801.03783.pdf
    # This is the most polished version available


    M_C = 10000


    ###Population Part

    pop_list = list(partition["population"].values())
    pop_ideal = np.sum ( pop_list) / 13
    J_p = np.linalg.norm(( [ x/ pop_ideal - 1 for x in pop_list]))


    ###PolsbyPopperPart

    J_I = np.sum( list( partition["polsby_popper"]))

    ###County Splits

    df["current"] = df[unique_label].map(partition.assignment)
    county_splits = {} #Maps to list of number of VTDS in each districts

    for pair in df.groupby(["COUNTYFP10", "current"])["GEOID10"]:
        county_splits[pair[0][0]] = []

    for pair in df.groupby(["COUNTYFP10", "current"])["GEOID10"]:
        county_splits[pair[0][0]].append(len(pair[1]))

    # Checking against num splits: len ( [x for x in county_splits.keys() if len(county_splits[x]) > 1] )

    num_2_splits = 0
    num_2_splits_W = 0
    num_greater_splits = 0
    num_greater_splits_W = 0


    for county in county_splits.keys():
        if len( county_splits[county]) == 2:
            total = sum( county_splits[county])
            max_2 = min( county_splits[county])
            num_2_splits += 1
            num_2_splits_W += np.sqrt( max_2 / total )
        if len(county_splits[county]) > 2:
            total = sum(county_splits[county])
            county_splits[county].sort()
            left_overs = total - county_splits[county][-1] - county_splits[county][-2]
            num_greater_splits += 1
            num_greater_splits_W += np.sqrt( left_overs / total)

    J_c = num_2_splits * num_2_splits_W + M_C * num_greater_splits * num_greater_splits_W

    ##VRA score --> @Daryl


    return score



def num_splits(partition, df):
    df["current"] = df.index.map(dict(partition.assignment))
    return sum(df.groupby("COUNTYFP10")["current"].nunique() > 1)






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
    temperature = 100000
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
        total_steps=10000
    )

    for part in chain:
        pass
        #print(deviation(list(part["population"].values())))

    initial_partition = GeographicPartition(
    graph,
    assignment= part.assignment,
    updaters={
        "polsby_popper" : polsby_popper,
        "cut_edges": cut_edges,
        "population": Tally("TOT_POP", alias="population"),
        "SEN12": election
    })

    '''
    values = []
    for part in chain:
        values.append ( [deviation(list(part["population"].values())), np.mean(list(part["polsby_popper"].values()))])

    print('started', values[0])
    print('ended', values[-1])
    '''

    return initial_partition

def run_without_constraints(starting_partition):

    chain = MarkovChain(
        proposal=propose_random_flip,
        constraints=[single_flip_contiguous],
        accept=always_accept,
        initial_state=starting_partition,
        total_steps=10000
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



def getting_started():


    graph = Graph.from_file("./PA_VTD.shp")

    election = Election("SEN12", {"Dem": "USS12D", "Rep": "USS12R"})

    starting_partition = GeographicPartition(
        graph,
        assignment="2011_PLA_1",
        updaters={
            "polsby_popper" : polsby_popper,
            "cut_edges": cut_edges,
            "population": Tally("TOT_POP", alias="population"),
            "SEN12": election,
            "County Splits": num_splits
        }
    )

    df = gpd.read_file("./PA_VTD.shp")

    '''
    initial_partition = Partition(
        graph,
        assignment="2011_PLA_1",
        updaters={
            "cut_edges": cut_edges,
            "population": Tally("TOT_POP", alias="population"),
            "SEN12": election
        }
    )
    '''

def mattingly_2018_annealing(starting_partition):

    chain = MarkovChain(
        proposal=propose_random_flip,
        constraints=[single_flip_contiguous],
        accept=always_accept,
        initial_state=starting_partition,
        total_steps=40000
    )
    for part in chain:
        pass

    ## Use an updated to  grow beta linearly over 60,000 accepted steps
    ## -> @Daryl

    ## beta fixed at one for 20000 *accepted* steps

    #At the end of this, we get one sample

    return plan


# def mattingly_2018_threshold(plans):



def test_annealing(starting_partition):

    stages = 5

    annealed_partitions = []

    for k in range(stages):
        starting_partition = run_without_constraints(starting_partition)
        annealed_partition = run_with_MCMC_constraints(starting_partition)
        print( "unnealed", [deviation(list(starting_partition["population"].values())), np.mean(list(starting_partition["polsby_popper"].values()))] )
        print(  "annealed", [deviation(list(annealed_partition["population"].values())), np.mean(list(annealed_partition["polsby_popper"].values()))] )
        annealed_partitions.append(annealed_partition)
        starting_partition = annealed_partition


    return True