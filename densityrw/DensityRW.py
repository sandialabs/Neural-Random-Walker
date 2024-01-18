# Copyright 2023 National Technology & Engineering Solutions of Sandia, LLC (NTESS). 
# Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains 
# certain rights in this software.
# Apache License Version 2.0, January 2004 http://www.apache.org/licenses/
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import time
from fugu import input_coding_types
from fugu.bricks import Brick, Vector_Input
from fugu.scaffold import Scaffold
from fugu.backends import snn_Backend
import networkx as nx
import numpy as np
import ast
from scipy.io import loadmat
from numpy import genfromtxt

import logging

rw_logger = logging.getLogger("NRW")

test_timesteps = 50
num_test_nodes = 25

import densityrw.SpikingPDE as SpikingPDE


class DensityRW(Brick):
    """Basic Density Random Walk Circuit
    
    Arguments:
        + timesteps - An integer number of timesteps to run the walk
        + transitions - A dictionary of valid transition edges on the graph {source_node:neighbors}, 
                        neighbors should be a list of tuples (destination_node, probability)
        + init_walkers - A dictionary {start_node: number_of_walkers} to initialize the walk
    
    """
    def __init__(self, 
                 timesteps, 
                 transitions, 
                 init_walkers={(10,):30, (13,): 30}, 
                 name=None, 
                 coding='Raster'):
        super(Brick, self).__init__()
        self.is_built = False
        self.metadata = {}
        self.output_coding = coding
        self.supported_codings = input_coding_types
        self.name = name        
        self.brick_tag = 'DensityRW'
        self.net = None
        self.graph = None
        self.injection = None
        self.transitions = transitions
        self.init_walkers = init_walkers
        self.timesteps = timesteps
    
    def build(self,
              graph,
              metadata,
              control_nodes,
              input_lists,
              input_codings):
        """
        Build Density RW brick.

        Arguments:
            + graph - networkx graph to define connections of the computational graph
            + metadata - dictionary to define the shapes and parameters of the brick
            + control_nodes - dictionary of lists of auxillary networkx nodes.  Expected keys: 'complete' - A list of neurons that fire when the brick is done
            + input_lists - list of nodes that will contain input
            + input_coding - list of input coding formats.  All codings are allowed.

        Returns:
            + graph of a computational elements and connections
            + dictionary of output parameters (shape, coding, layers, depth, etc)
            + dictionary of control nodes ('complete')
            + list of output (1 output)
            + list of coding formats of output (Coding matches input coding)
        """      
        #Basic Checks
        if len(input_lists) != 1:
            raise ValueError('Incorrect Number of Inputs.')
        for input_coding in input_codings:
            if input_coding not in self.supported_codings:
                raise ValueError("Unsupported Input Coding. Found: {}. Allowed: {}".format(input_coding,
                                                                                           self.supported_codings))       
        
        #Complete Node
        control_node_list = []
        control_node_name = self.name + '_complete'
        graph.add_node(control_node_name,
                       index=(-1,),
                       decay=0.0,
                       potential=0.0,
                       threshold = float(self.timesteps)
                )
        graph.add_edge(control_nodes[0]['complete'], control_node_name, weight=1.0, delay=1)
        timestep_counter_name = self.name + '_timestep_counter'
        graph.add_node(timestep_counter_name,
                       index=(-2,),
                       decay = 0.0,
                       potential = 0.0,
                       threshold = 0.5
                       )
        graph.add_edge(timestep_counter_name, timestep_counter_name, weight=1.0, delay=1)
        graph.add_edge(timestep_counter_name, control_node_name, weight=1.0, delay=1)
        
        
        #Build the walking graph
        self.net = SpikingPDE.MarkovNetwork(initial_walkers=self.init_walkers,
                            transitions=self.transitions,
                            synchronized=True,
                            log_potential=True, log_spikes=True)
        self.graph = self.net.build()
        self.graph = SpikingPDE.to_graph(self.graph, SpikingPDE.neuron_list)
        
        
        #Prep initial position of walkers
        counter_neurons = [node for node in self.graph.nodes if 'counter' in self.graph.nodes[node]['groups']]
        for init_source in self.init_walkers:
            for node in counter_neurons:
                if str(init_source) in self.graph.nodes[node]['groups']:
                    self.graph.nodes[node]['potential']= -self.init_walkers[init_source]
        
        #Rename nodes
        relabel_dictionary = {}
        for node in self.graph.nodes:
            relabel_dictionary[node] = self.name + '_' + str(node) + '_groups_' + str(self.graph.nodes[node]['groups'])
        new_supervisor_walker = relabel_dictionary[self.net.walker_supervisor.name]
        self.graph = nx.relabel_nodes(self.graph, relabel_dictionary, copy=False)
        
        #Grab references to output nodes
        output_nodes = [node for node in self.graph.nodes if 'readout' in self.graph.nodes[node]['groups']]
        
        #Write indices for outputs (also bad code)
        for node in output_nodes:
            self.graph.nodes[node]['index'] = ast.literal_eval(self.graph.nodes[node]['groups'][1])
        
        #Add RW nodes to Fugu graph (This is potentially memory intensive)
        graph = nx.compose(graph, self.graph)
        
        #Start the walk
        graph.add_edge(control_nodes[0]['complete'], new_supervisor_walker, weight=5.0, delay=1.0)

        self.is_built = True
        return (graph, self.metadata, [{'complete':control_node_name}], [output_nodes], [self.output_coding])
    

def build_test_graph(num_nodes=20):
    transitions = {}
    for i in range(0,num_nodes-1):
        transitions[(i,)] = SpikingPDE.Transition(location=(i,), neighbors=[((i+1,),0.5)])
    transitions[(num_nodes-1,)] = SpikingPDE.Transition(location=(9,),neighbors=[])
    for i in range(0,num_nodes-1):
        transitions[(i+1,)].neighbors.append(((i,), 0.5))
    transitions[(0,)].neighbors.append(((num_nodes-1,),0.5))
    transitions[(num_nodes-1,)].neighbors.append(((0,),0.5))
    return transitions
    
def build_network(num_nodes=20, init_walkers={(10,):30, (13,):30}):
    transitions = build_test_graph(num_nodes)
    net = SpikingPDE.MarkovNetwork(initial_walkers=init_walkers,
                        transitions=transitions,
                        synchronized=True,
                        log_potential=True, log_spikes=True)
    graph = net.build()
    graph = SpikingPDE.to_graph(graph, SpikingPDE.neuron_list)
    return net, graph, init_walkers

def time_spiking_PDE(net, runtime):
    start_time = time.time()
    net.run(runtime)
    end_time = time.time()
    return end_time - start_time

def load_transitions(mat_file,
                remove_sink_connections=True,
                verbose=1):
    SpikingPDE.reset_neuron_list()
    if '.mat' in mat_file:
        prob_mtx = loadmat(mat_file)
        prob_mtx = prob_mtx[mat_file[:-4]]
    elif '.csv' in mat_file:
        prob_mtx = genfromtxt(mat_file, delimiter=',')
    else:
        raise ValueError("Matrix file incompatible")
    N = np.shape(prob_mtx)
    transitions = {}
    for i in range(N[0]):
        neighbors = []
        p = 0
        for j in range(N[1]):
            prob_i_j = prob_mtx[i,j]
            p += prob_i_j
            if prob_i_j > 0:
                neighbors.append(((j,), prob_mtx[i,j]))
        if verbose>0:
            rw_logger.info("node " + str(i) + " has total probability of " + str(p) )
        if remove_sink_connections and len(neighbors)==1 and neighbors[0]==((i,),1.0):
            if verbose>0:
                rw_logger.info("Removing connections from a sink at " + str((i,)))
            neighbors=[]
        transitions[(i,)] = SpikingPDE.Transition(location=(i,), neighbors=neighbors)
    return transitions

def build_fugu_network(num_nodes = 20, 
                       init_walkers={(10,): 5, (13,): 5}):
    transitions = build_test_graph(num_nodes)
    scaffold = Scaffold()
    scaffold.add_brick(Vector_Input(np.array([1]), coding='Raster', name='Input0'), 'input' )
    scaffold.add_brick(DensityRW(test_timesteps, transitions, init_walkers=init_walkers, name='DensityRW'), [0], output=True)
    scaffold.lay_bricks()
    return scaffold

def time_fugu(scaffold, runtime):
    start_time = time.time()
    backend = snn_Backend()
    backend.compile(scaffold)
    for node in scaffold.graph.nodes:
      if 'readout' in node:
        rw_logger.info(scaffold.graph.nodes[node])
    result = backend.run(runtime)
    end_time = time.time()
    return end_time - start_time, scaffold, result
    
def run_test():
    rw_logger.info('Running old build process...')
    net, graph, init_walkers = build_network(num_test_nodes)
    rw_logger.info('Runtime: {} seconds'.format(time_spiking_PDE(net, test_timesteps)))
    rw_logger.info('Running Fugu build process...')
    fugu_time, scaffold, result = time_fugu(build_fugu_network(num_test_nodes), test_timesteps)
    rw_logger.info('Runtime: {} seconds'.format(fugu_time))
    rw_logger.info(result)
    return scaffold
    
if __name__ == "__main__":
    scaffold = run_test()
