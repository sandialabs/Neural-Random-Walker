# Copyright 2023 National Technology & Engineering Solutions of Sandia, LLC (NTESS). 
# Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains 
# certain rights in this software.
# Apache License Version 2.0, January 2004 http://www.apache.org/licenses/
from __future__ import print_function
#Basic Imports
import numpy as np
import scipy as sp
import networkx as nx
from collections import deque

import logging

#Config Imports
from .SpikingConfig import max_value 

rw_logger = logging.getLogger("NRW")

#Spiking Backend
#Global Variables
num_neurons = 0
neuron_list = []
spike_list = deque() #[]

def to_graph(graph, tmp_neuron_list):
    graph_to_return = graph.copy()
    for neuron in tmp_neuron_list:
        node = graph_to_return.nodes[neuron.name]
        for var in vars(neuron):
            node[var] = vars(neuron)[var]
    graph_to_return.graph['has_delay'] = True
    return graph_to_return

# Resets an internal list of neuron_list
# This is required if you want start a new network once a network has been created
# A better implementation would be to have a 'neuron manager' object
def reset_neuron_list():
    global num_neurons
    num_neurons = 0
    global neuron_list
    neuron_list = []
    global spike_list
    spike_list = deque() #[]

# Returns a list of all neurons that belong to the given group.
# The group should be a string, though strictly speaking it could be any type of hashable object
def get_neuron_group(group_name):
    global neuron_list
    return [n for n in neuron_list if group_name in n.groups]
# Returns a list of all neurons that match the given name
# Names should be unique, though this is not enforced.
# As such, some downstream functions assume a list of length 1 is returned by this function
# The name should be a string, though strictly speaking it could be any time of hashable object
def get_neuron_from_name(name):
    global neuron_list
    return [n for n in neuron_list if n.name==name]

# Runs the network one timestep forward.
# In order, the following operations are performed
# 1) Spikes in flight are routed.  Spikes that are received at the current timestep
# are integrated into the corresponding postsynaptic neurons.  Spikes that are still
# in flight are updated to be one timestep closer.
# 2) Dead spikes are removed from the system
# 3) Each neuron (node in the graph) has its potential compared against the threshold
#  a) If the threshold is exceeded, a random number is compared against the probability to fire.
#  b) If the neuron fires, then spikes are entered into an array (spike_list) with the appropriate postsynaptic neuron, weight, and delay
#  c) If a neuron does not fire, its potential is reset according to reset_mode
#  d) If the neuron does not fire, its potential is updated according to the decay constant
#
# Thi s function supports two logging mechanisms spike_log and potential_log
# potential_log is simply a dictionary where each key is a timestep and each value is a list of potentials
# spike_log is a list of spike events.  Spike events use the following coding.
# Spike_log Coding:
# Tuple (neuron, timestep, code)
# 0 = Threshold exceeded
# 1 = Spike sent
# 2 = Spike received
# For code 2, there are two additional pieces of information
# event[3] = presynaptic neuron
# event[4] = weight
def update_network(network, spike_log = None, potential_log = None, timestep = 0, log_mode='normal'):
    global spike_list
    global neuron_list
    spikes_to_remove = deque()
    for spike in spike_list:
        if(spike.delay>1):
            spike.delay = spike.delay-1
        elif(spike.delay==1):
            if(spike_log is not None and log_mode == 'normal'):
                spike_log.append((spike.postsynaptic, timestep, 2, spike.presynaptic, spike.weight))
            spike.postsynaptic.potential+= spike.weight
            spikes_to_remove.append(spike)
        else:
            rw_logger.info("MALFORMED SPIKE!")
    for spike in spikes_to_remove:
        spike_list.remove(spike)
    for node in network.nodes:
        neurons = get_neuron_from_name(node)
        for neuron in neurons:
            if neuron.potential > neuron.threshold:
                if(spike_log is not None):
                    spike_log.append((neuron, timestep, 0))
                r = np.random.rand()
                if(r < neuron.p):
                    if(spike_log is not None):
                        spike_log.append((neuron, timestep, 1))
                    neighbors = network.neighbors(neuron.name)
                    for neighbor in neighbors:
                        spike_list.append(Spike(neuron, get_neuron_from_name(neighbor)[0], network[node][neighbor]['weight'], network[node][neighbor]['delay'] ))
                    neuron.potential = 0
                if(neuron.reset_mode=='threshold'):
                    neuron.potential = 0
            else:
                neuron.potential = neuron.potential*(1-neuron.decay)
    if potential_log is not None:
        potential_log[timestep] = [neuron.potential for neuron in neuron_list]
    return (timestep, spike_log, potential_log)

#Spiking Backend
#A class for Neuron Objects
#Parameters are set using keywords
#Parameters are name, group, threshold, decay, p, reset_mode
#At initialization group should be a single hashable Objects
#However, the group is assigned as the first element in a list groups
#This allows a neuron to belong to more than one group
#p is the probability of spiking when the threshold is reached
# reset_mode has two options:  'spike' = reset after a spike
# 'threshold' - reset after a threshold is reached
class Neuron:
    def __init__(self,**kwargs):
        global num_neurons
        self.name = num_neurons
        if 'name' in kwargs:
            self.name = kwargs['name']
        self.groups = []
        if 'group' in kwargs:
            self.groups = [kwargs['group']]
        self.threshold = 1
        if 'threshold' in kwargs:
            self.threshold = kwargs['threshold']
        self.decay = 1
        if 'decay' in kwargs:
            self.decay = kwargs['decay']
        self.p = 1
        if 'p' in kwargs:
            self.p = kwargs['p']
        self.reset_mode = 'spike'
        ##reset_mode options are
        ## 'spike' - reset after a spike
        ## 'threshold' - reset after a threshold is reached
        if 'reset_mode' in kwargs:
            self.reset_mode = kwargs['reset_mode']
        self.potential = 0
        num_neurons = num_neurons+1
        global neuron_list
        neuron_list.append(self)

#Spiking Backend
#A class for Spike Objects
#Spikes should not be instantiated directly
class Spike:
    def __init__(self, presynaptic, postsynaptic, weight, delay):
        self.presynaptic = presynaptic
        self.postsynaptic = postsynaptic
        self.delay = delay
        self.weight = weight

#Markov Backend
#A Unit represents a discrete area in space through which walkers pass.
#Walkers may occupy only one unit
#Unless in transit, walkers must occupy exactly one unit
# Units are initialized with coordinates (a tuple of user-defined values), a max_value (which is the maximum value represented by the readout neuron), and probability_bits (which is the number of probability gates)
# Methods:
# connect - Connects one unit to the passed unit with specified probability
# add_to_graph - Adds neurons and edges to the specified graph.  Units should be connected before being added to a add_to_graph
class Unit:
    def __init__(self, coordinates, max_value, probability_bits, readout=None, synchronized=True):
        self.coordinates = coordinates
        self.synchronized = synchronized
        self.max_value = 1.0*max_value
        if readout is None:
            self.readout_neuron = Neuron(group='readout', decay=0, threshold=0.5)
        else:
            self.readout_neuron = readout
        self.walker_counter_neuron = Neuron(group='counter', decay=0, threshold=0.0)
        self.walker_generator_neuron = Neuron(group='generator', threshold=0.5)
        self.random_gates = []
        for i in range(0, probability_bits):
            self.random_gates.append(Neuron(group='random_gate', threshold=0.5, reset_mode='threshold'))
        self.probabilities = np.zeros((probability_bits+1,))
        self.neighbors = []
        self.output_gate_neurons = []
        for i in range(0, probability_bits+1):
            self.output_gate_neurons.append(Neuron(group='output_gate', reset_mode='threshold', threshold=0.5))
        self.neurons = [self.readout_neuron, self.walker_counter_neuron, self.walker_generator_neuron]
        self.neurons.extend(self.random_gates)
        self.neurons.extend(self.output_gate_neurons)
        if self.synchronized:
            self.buffer = Neuron(group='buffer', decay = 0, threshold=0.0)
            self.buffer_control = Neuron(group='generator', threshold=0.5)
            self.neurons.extend([self.buffer, self.buffer_control])
        for neuron in self.neurons:
            neuron.groups.append(str(coordinates))


    def connect(self, target_unit, probability):
        if(len(self.neighbors) < len(self.probabilities)):
            self.probabilities[len(self.neighbors)] = probability
            self.neighbors.append(target_unit)
        else:
            rw_logger.info("Out of space for neighbors!  Allocate more using probability_bits!")

    def add_to_graph(self, graph, walker_supervisor=None, walks_complete=None, simulation_supervisor=None, buffer_supervisor=None, buffer_clear=None):
        for neuron in self.neurons:
            graph.add_node(neuron.name,label=neuron.name)
        if self.synchronized:
            graph.add_edge(self.buffer_control.name, self.walker_counter_neuron.name, weight = -1.0, delay=1)
            graph.add_edge(self.buffer_control.name, self.buffer_control.name, weight =1.0, delay=1)
            graph.add_edge(self.buffer_control.name, self.buffer.name, weight=1.0, delay=1)
            graph.add_edge(self.buffer.name, self.walker_counter_neuron.name, weight = 1.0, delay=1)
            graph.add_edge(self.buffer.name, self.buffer_control.name, weight=-1.0, delay=1)
            graph.add_edge(self.buffer.name, self.buffer.name, weight=-1.0, delay=1)
            graph.add_edge(self.buffer_control.name, self.walker_counter_neuron.name, weight=-1.0, delay=1)
            if buffer_supervisor is not None:
                graph.add_edge(buffer_supervisor.name, self.buffer.name, weight=1.0, delay=1)
                graph.add_edge(buffer_supervisor.name, self.buffer_control.name, weight=1.0, delay = 1)
            if buffer_clear is not None:
                graph.add_edge(self.buffer.name, buffer_clear.name, weight=1.0, delay=1)
        graph.add_edge(self.walker_generator_neuron.name,  self.readout_neuron.name, weight=1.0, delay=1)
        graph.add_edge(self.walker_generator_neuron.name, self.walker_generator_neuron.name, weight=1.0, delay = 1)
        graph.add_edge(self.walker_generator_neuron.name, self.walker_counter_neuron.name, weight=1.0, delay=1)
        graph.add_edge(self.walker_counter_neuron.name, self.readout_neuron.name, weight=-1.0, delay=1)
        graph.add_edge(self.walker_counter_neuron.name, self.walker_counter_neuron.name, weight=-1.0, delay=1)
        graph.add_edge(self.walker_counter_neuron.name, self.walker_generator_neuron.name, weight=-1.0, delay=1)
        self.moderator_values = np.zeros(len(self.output_gate_neurons))

        output_length = len(self.output_gate_neurons)
        self.random_gates[0].p = self.probabilities[0]
        for i in range(0,output_length-1):
            self.random_gates[i].p = self.probabilities[i]/(1 - sum(self.probabilities[0:i]))
        
        for i in range(0, output_length-1):
            graph.add_edge(self.walker_generator_neuron.name, self.random_gates[i].name, weight=1.0, delay=i+1)
            graph.add_edge(self.walker_counter_neuron.name, self.random_gates[i].name, weight=-1.0, delay=i+1)
            graph.add_edge(self.random_gates[i].name, self.output_gate_neurons[i].name, weight=1.0, delay=1)
            k = 1
            for j in range(i+1, output_length-1):
                graph.add_edge(self.random_gates[i].name, self.random_gates[j].name, weight=-1.0, delay=k)
                k += 1
            graph.add_edge(self.random_gates[i].name, self.output_gate_neurons[-1].name, weight=-1.0, delay=k)
        graph.add_edge(self.walker_generator_neuron.name, self.output_gate_neurons[-1].name, weight=1.0, delay=output_length)
        graph.add_edge(self.walker_counter_neuron.name, self.output_gate_neurons[-1].name, weight=-1.0, delay=output_length)
        
        for i in range(0, len(self.neighbors)):
            neighbor = self.neighbors[i]
            gate = self.output_gate_neurons[i]
            input_neuron = neighbor.walker_counter_neuron
            if neighbor.synchronized:
                input_neuron = neighbor.buffer
            graph.add_edge(gate.name, input_neuron.name, weight=-1.0, delay=1)
        if walker_supervisor is not None:
            graph.add_edge(walker_supervisor.name, self.walker_counter_neuron.name, weight=1.0, delay=1)
            graph.add_edge(walker_supervisor.name, self.walker_generator_neuron.name, weight=1.0, delay=1)
        if simulation_supervisor is not None:
            graph.add_edge(simulation_supervisor.name, self.readout_neuron.name, weight=1.0, delay=1)
        if walks_complete is not None:
            graph.add_edge(self.walker_counter_neuron.name, walks_complete.name, weight=1.0, delay=len(self.probabilities))

#Represents the transition matrix from one unit to its neighbors.
#'location' is a user-defined key for the origin unit's location
#A standard choice is a tuple of coordinates e.g. (1,2)
#'neighbors' is a list of tuples [(destination.location, probability),... ]
class Transition:
    def __init__(self, **kwargs):
        if 'location' in kwargs:
            self.location = kwargs['location']
        self.neighbors = []
        if 'neighbors' in kwargs:
            self.neighbors = kwargs['neighbors']

#A spiking network that simulates random Markov walks
#'transitions' is a dictionary {location : transition }
#'initial_walkers' is a dictionary {location : number of walkers}
#'log_potential' is boolean, if True, the potentials are logged according to run_simulation
# self.potential_log is otherwise None
#'log_spikes' is boolean, if True, the potentials are logged according to run_simulation
# self.spike_log is otherwise None
#'synchronized' is a boolean.  If true, walkers and stored in a buffer before being sent to
# a new unit.  This ensures that each walker takes the same number of steps.
class MarkovNetwork:
    def __init__(self,**kwargs):
        self.transitions = {}
        if 'transitions' in kwargs:
            self.transitions = kwargs['transitions']
        self.initial_walkers = {}
        if 'initial_walkers' in kwargs:
            self.initial_walkers = kwargs['initial_walkers']
        self.log_potential = False
        self.potential_log = None
        if 'log_potential' in kwargs:
            self.log_potential = kwargs['log_potential']
        self.log_spikes = False
        self.spike_log = None
        if 'log_spikes' in kwargs:
            self.log_spikes = kwargs['log_spikes']
        self.synchronized = True
        if 'synchronized' in kwargs:
            self.synchronized = kwargs['synchronized']
        self.built = False
    def build(self):
        self.num_units = len(self.transitions)
        self.graph = nx.DiGraph()
        reset_neuron_list()
        self.all_units = []
        for location in self.transitions:
            neighbors = self.transitions[location].neighbors
            ##EDIT  len(neighbors)-1 -> max(len(neighbors)-1, 1)
            self.all_units.append(Unit(location, max_value, max(len(neighbors)-1,1), synchronized=self.synchronized))
        for unit in self.all_units:
            for (neighbor, p) in self.transitions[unit.coordinates].neighbors:
                matching_units = [unit for unit in self.all_units if unit.coordinates==neighbor]
                if len(matching_units)<1:
                    rw_logger.info("Something went wrong in finding neighbors.... At least one neighbor is missing")
                    rw_logger.info("Available Coordinates")
                    rw_logger.info([unit.coordinates for unit in self.all_units])
                    rw_logger.info("Needed Coordinates")
                    rw_logger.info(neighbor)
                if len(matching_units)>1:
                    rw_logger.info("Something went wrong in finding neighbors.... Too many neighbors found")
                    rw_logger.info("Available Coordinates")
                    rw_logger.info([unit.coordinates for unit in self.all_units])
                    rw_logger.info("Needed Coordinates")
                    rw_logger.info(neighbor)
                unit.connect(matching_units[0], p)
        #Control neurons
        self.walker_supervisor = Neuron(threshold=1.0, decay=1.0, group='controller')
        self.simulation_supervisor = Neuron(threshold=100.0, decay=1.0, group='controller')
        self.walks_complete = Neuron(threshold=self.num_units-0.5,decay=0.0, group='controller')
        self.buffer_supervisor = None
        self.buffer_clear = None
        if self.synchronized:
            self.buffer_supervisor = Neuron(threshold=1.0, decay=1.0, group='controller')
            self.buffer_supervisor.groups.append('buffer-control')
            self.buffer_clear = Neuron(threshold=self.num_units-0.5, decay=0.0, group = 'controller')
            self.buffer_clear.groups.append('buffer-control')

        self.injection = {}
        self.injection[0] = [(self.walker_supervisor,10)]

        for unit in self.all_units:
            if unit.coordinates in self.initial_walkers:
                self.injection[0].append((unit.walker_counter_neuron,-self.initial_walkers[unit.coordinates]))

        #Add units to graph
        for unit in self.all_units:
            unit.add_to_graph(self.graph, walker_supervisor=self.walker_supervisor, walks_complete=self.walks_complete, simulation_supervisor=self.simulation_supervisor, buffer_supervisor=self.buffer_supervisor, buffer_clear=self.buffer_clear)

        if self.synchronized:
            self.graph.add_edge(self.buffer_clear.name, self.walker_supervisor.name, weight=2.0, delay=2)
            self.graph.add_edge(self.walks_complete.name, self.buffer_supervisor.name, weight=2.0, delay=2)
        else:
            self.graph.add_edge(self.walks_complete.name, self.walker_supervisor.name, weight=2.0, delay=2)


        if self.log_potential:
            self.potential_log = np.zeros((0, len(neuron_list)))
            self.potential_log[:] = np.NaN
        if self.log_spikes:
            self.spike_log = deque()

        self.built = True
        return self.graph
    def run(self, runtime, **kwargs):
        if 'verbose' in kwargs:
            verbose = kwargs['verbose']
        else:
            verbose = 0
        if 'log_mode' in kwargs:
            log_mode = kwargs['log_mode']
        else:
            log_mode = 'short'
        if not self.built:
            self.build()
        tmp_log = np.zeros((runtime, len(neuron_list)))
        tmp_log[:] = np.NaN
        self.potential_log = np.append(self.potential_log, tmp_log, axis=0)
        (spike_log, potential_log) = run_simulation(runtime, self.graph, injection=self.injection, spike_log=self.spike_log, potential_log = self.potential_log, verbose=verbose, log_mode=log_mode)
        return (spike_log, potential_log)



#Runs the markov simulation for a given number of time steps
# spike_log and potential_log are documented above
# injection is a dictionary which manages controlled current injection {timestep: [(neuron, current),]}
def run_simulation(runtime, network, injection={}, spike_log = None, potential_log =None, **kwargs):
    if 'verbose' in kwargs:
        verbose = kwargs['verbose']
    else:
        verbose = 0
    if 'log_mode' in kwargs:
        log_mode = kwargs['log_mode']
    else:
        log_mode = False
    for timestep in range(0,runtime):
        if verbose==1:
            rw_logger.debug(str(timestep+1)+"/"+str(runtime)+"       ", end='\r')
        if timestep in injection:
            for (neuron, current) in injection[timestep]:
                neuron.potential += current
        update_network(network, timestep = timestep, spike_log = spike_log, potential_log=potential_log, log_mode=log_mode)
    return (spike_log, potential_log)
