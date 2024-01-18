# Copyright 2023 National Technology & Engineering Solutions of Sandia, LLC (NTESS). 
# Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains 
# certain rights in this software.
# Apache License Version 2.0, January 2004 http://www.apache.org/licenses/
import math
import numpy as np
import nxsdk.api.n2a as nx
import networkx as networkx
from nxsdk.graph.monitor.probes import *

import logging

from .rw_core import *

rw_logger = logging.getLogger("NRW")

def define_inputs(net, time_delay, num_walkers, input_times = [1]):
    #### DEFINE INPUTS ####

    # Create Spike Generation Process
    spike_inputs = net.createSpikeGenProcess(numPorts = 2)
    st=[]

    for i in input_times:
        s_in=list(range(i, num_walkers+i))
        st=st+s_in

    rw_logger.info("Number of inputs: {} with delay: {}".format( len(st), time_delay) )

    spike_inputs.addSpikes(spikeInputPortNodeIds = 0, spikeTimes=st)
    spike_inputs.addSpikes(spikeInputPortNodeIds = 1, spikeTimes=[time_delay+3])
    return spike_inputs


def probRandomizeCircuit(net, core_prob, N_random=4):
    ####
    # Circuit is placed on each stochastic core to "burn in" some variable degree of RNG calls to improve pseudo randomness
    #
    # These neurons do not do anything for the model, but effectively move random seed around.

    weight_precision_factor=1
    proto_prob=nx.CompartmentPrototype(biasMant=10,
                          biasExp=6,
                          vThMant=800,
                          compartmentVoltageDecay = 100,
                          compartmentCurrentDecay = 100,
                          enableNoise=1,
                          logicalCoreId=core_prob,
                          functionalState=nx.COMPARTMENT_FUNCTIONAL_STATE.IDLE,
                          randomizeVoltage=1,
                          noiseMantAtCompartment=0, # Threshold = center around 0
                          noiseExpAtCompartment=12+math.log(weight_precision_factor,2))
    connProto = nx.ConnectionPrototype(signMode=nx.SYNAPSE_SIGN_MODE.MIXED)
    weights=np.random.randint(2, size=(N_random, N_random))*100
    mask=weights/100
    random_neurons = net.createCompartmentGroup(size=N_random, prototype=proto_prob)
    random_conns=random_neurons.connect(random_neurons, prototype=connProto, delay=np.array([3]), weight=weights, connectionMask=mask)

    return random_neurons, random_conns

def connect_mesh(mesh, counter_weight):
    #####
    # Connects output neurons of each mesh point to appropriate buffer input neurons.
    #

    connProto_In = nx.ConnectionPrototype(signMode=nx.SYNAPSE_SIGN_MODE.INHIBITORY)
    verbose=0
    a=(list(mesh.edges))
    ct = 0
    k=0
    for i in mesh.nodes:
        mesh.nodes[i]['output_conns']=[]
        ct=ct+k
        k=0
        k1=0
        print( (ct+0.0)/len(a)/4, end='\r')
        adj_list=list(sorted(mesh.adj[i].keys()))
        weights=mesh.adj[i]
        weights_simple=[]
        for output in mesh.adj[i].keys():
            if('weight' in weights[output]):
                weights_simple.append([weights[output]['weight'], output])
            else:
                weights_simple.append([.25, output])
        weight_ids= list(sorted(weights_simple))

        num_outputs=len(mesh.adj[i].keys())
        if(num_outputs<4):
            num_outputs=4;

        if(verbose):
            rw_logger.info("i: {}, k: {}, weight_ids: {}, num_outputs: {}".format(i, k, weight_ids, num_outputs))

        for conn in weight_ids:
            j=conn[1]
            weight_mat=np.zeros((3,num_outputs))
            weight_mat[0, k]=-counter_weight
            weight_mask=np.zeros((3,num_outputs))
            weight_mask[0, k]=1
            if(mesh.nodes[i]['chip']!=mesh.nodes[j]['chip']):
                mesh.nodes[i]['output_conns'].append(mesh.nodes[i]['outputs'].connect(mesh.nodes[j]['buffer'], prototype=connProto_In, weight=weight_mat, connectionMask= weight_mask))
                k1=1
            k=k+1
    for i in mesh.nodes:
        ct=ct+k
        k=0
        print((ct+0.0)/len(a)/4, end='\r')

        adj_list=list(sorted(mesh.adj[i].keys()))

        weights=mesh.adj[i]

        weights_simple=[]
        for output in mesh.adj[i].keys():
            if('weight' in weights[output]):
                weights_simple.append([weights[output]['weight'], output])
            else:
                weights_simple.append([.25, output])
        weight_ids= list(sorted(weights_simple))

        num_outputs=len(mesh.adj[i].keys())
        if(num_outputs<4):
            num_outputs=4

        for conn in weight_ids:
            j=conn[1]
            weight_mat=np.zeros((3,num_outputs))
            weight_mat[0, k]=-counter_weight
            weight_mask=np.zeros((3,num_outputs))
            weight_mask[0, k]=1
            if(mesh.nodes[i]['chip']==mesh.nodes[j]['chip']):
                mesh.nodes[i]['output_conns'].append(mesh.nodes[i]['outputs'].connect(mesh.nodes[j]['buffer'], prototype=connProto_In, weight=weight_mat, connectionMask= weight_mask))
            k=k+1



def makeNetwork(superNet,
                x,
                y,
                custom_network,
                core_density=9.0,
                input_x=[(0,0), (0,1), (0,2), (1,0), (1,1), (1,2)],
                core_start=0,
                sim_time=10000,
                num_walkers=100,
                track=0,
                verbose = 0,
                input_times=[1],
                s = 0,
                spikeStreamer = 0):
    rw_logger.info('    Building mesh')

    rw_logger.info("    Run starting")
    rw_logger.info("        Num walkers: {}".format(num_walkers))
    rw_logger.info("        locations: {}".format(input_x))
    rw_logger.info("        time: {}".format( sim_time))
    track_probes=[]

    # Set different random seeds for following scripts
    #
    # Loihi does not allow direct access to random seeds on cores.  So we induce some differences by pre-burning in each stochastic core with a random circuit for a variable time delay.

    seed_time=np.random.randint(2424242)
    time_delay=seed_time%99
    time_delay=10

    counter_weight=4 #8

    # Different tracking conditions; Loihi speed is throttled by I/O, so we only read off what is needed
    # Track 1, 3: only read off end state of neurons
    # Track 2, 4, 5: Track from beginning
    if(track==1 or track==3):
        customSpikeProbeCond = SpikeProbeCondition(tStart=sim_time-2)
        customUProbeCond = IntervalProbeCondition(tStart=sim_time-2)
    if(track==2 or track==5 or track==4):
        customSpikeProbeCond = SpikeProbeCondition(tStart=1)
        customUProbeCond = IntervalProbeCondition(tStart=1)

    # Define basic connection prototypes
    connProto = nx.ConnectionPrototype(signMode=nx.SYNAPSE_SIGN_MODE.MIXED)
    connProto_Ex = nx.ConnectionPrototype(signMode=nx.SYNAPSE_SIGN_MODE.EXCITATORY)
    connProto_In = nx.ConnectionPrototype(signMode=nx.SYNAPSE_SIGN_MODE.INHIBITORY)
    mesh=networkx.DiGraph(np.genfromtxt(custom_network, delimiter=','))
    x = mesh.number_of_nodes()

    x_size=x
    y_size=y

    ##########################################
    ##### Embedding of mesh on Loihi Cores
    ##########################################

    # We need to pre-define how the mesh nodes will be deployed onto the Loihi cores.
    # NxNet does not performing any embedding since there are both stochastic and deterministic neurons.

    # List of cores used.  We actually will use two cores for each mesh point, one for deterministic neurons and one for stochastic neurons
    core_list=[]

    # Count of how many mesh points are on each (deterministic) core.  Will be used to handle supervisors
    core_meshcount=np.zeros((48640,), dtype=int)
    core_neuroncount=np.zeros((48640,), dtype=int)
    max_coreid=core_start
    chip_max=127
    chip_count = 1

    # We place all regular deterministic cores and then consolidate stochastic neurons onto a set of stochastic cores following...
    current_core=core_start+1
    current_core_ct=0
    core_density = core_density*10
    for i in mesh.nodes:
        # Loop through model mesh points
        #
        # NetworkX orders mesh nodes somewhat arbitrarily; we take advantage of this to load-balance the network

        # We assign mesh point to the next available Loihi core
        mesh.nodes[i]['core_id']=current_core
        current_core_ct+=1

        # If Loihi core fills up, we move to the next core
        if(current_core_ct==core_density):
            current_core+=1
            current_core_ct=0
            if(current_core%128==0):
               chip_count=chip_count+1

        # On some runs, there appeared to be an instability for multi-chip simulations that this code alleviates
        skip=0

        # If core has not been used yet, add to core_list
        if(mesh.nodes[i]['core_id'] not in core_list):
            core_list.append(mesh.nodes[i]['core_id'])

        # Track number of mesh points on Loihi core
        core_meshcount[mesh.nodes[i]['core_id']]=core_meshcount[mesh.nodes[i]['core_id']]+1

        # Track number of cores used so far
        if(mesh.nodes[i]['core_id']>max_coreid):
            max_coreid=mesh.nodes[i]['core_id']

        # Track number of chips used so far
        mesh.nodes[i]['chip']=math.floor(mesh.nodes[i]['core_id']/128)

    # Finished with deterministic nodes.  Now we prepare for stochastic nodes... these have to be on different cores

    # Identify first stochastic core (next Loihi core available)
    random_core_start=max_coreid+1
    max_coreid+=1

    # Stochastic neurons will be allocated *later* in mesh loading

    if(verbose):
        for i in mesh.nodes:
            rw_logger.info("    core_start: {}, i: {}, core_id: {}".format(core_start, i, mesh.nodes[i]['core_id']))

    ##############################################
    #### Set up inputs
    ##############################################

    rw_logger.info("    Setting up input times: {}".format(input_times) )
    spike_inputs = define_inputs(superNet, time_delay, num_walkers, input_times)



    ##############################################
    #### Set up supervisor circuit
    ##############################################

    # We'll count the supervisor activations in units of 16, with the threshold equal to all of the populated cores
    weight_super=16
    #threshold_super = weight_super*len(core_list)


    threshold_super = weight_super*chip_count


    # Create Global Supervisor Nodes
    supervisor_core=core_start

    # Create prototype for tracking cumulative activity on supervisor core

    total_counter_weight=2
    proto_counter_total=nx.CompartmentPrototype(vThMant=1000*total_counter_weight-.5*total_counter_weight,
                                                logicalCoreId=supervisor_core,
                                                compartmentVoltageDecay = 0,
                                                compartmentCurrentDecay = 4095)


    # Create 2 supervisor neurons (one to start buffers, one to start counters)
    #
    # Each supervisor neuron has a voltage threshold of 16*[# cores being used]

    proto_counter_super=nx.CompartmentPrototype(vThMant=threshold_super-2,
                                             logicalCoreId=supervisor_core,
                                             compartmentVoltageDecay = 0,
                                             compartmentCurrentDecay = 4095) # Threshold = 0

    proto_tg_super=nx.CompartmentPrototype(vThMant=threshold_super,
                                             logicalCoreId=supervisor_core,
                                             compartmentVoltageDecay = 4095,
                                             compartmentCurrentDecay = 4095) # Threshold = 0

    supervisor=superNet.createCompartmentGroup(size = 2, prototype=proto_counter_super)
    supervisor_findzero=superNet.createCompartmentGroup(size = 3, prototype=proto_tg_super)

    core_neuroncount[supervisor_core]+=4

    # Weakly connect buffer and counter supervisor neurons to one another to boost over threshold

    supervisor.connect(supervisor, prototype=connProto_Ex, weight=np.array([[0, 4], [4, 0]]), delay=np.array([2]))

    # These supervisor check neurons look to see if there are no walkers left
    supervisor.connect(supervisor_findzero, prototype=connProto_Ex, weight=np.array([[threshold_super+2, 0], [0, 0], [0, threshold_super*.5+2]]), delay=np.array([[5, 0], [0, 0], [0, 2]]))
    supervisor_findzero.connect(supervisor_findzero, prototype=connProto_Ex, weight=np.array([[0, 0, 0],[threshold_super+2, 0, 0], [0, threshold_super*.5, 0]]), delay=np.array([[0, 0, 0], [5, 0, 0], [0, 1, 0]]))

    rw_logger.info("Compartment index for supervisor_findzero is: {}".format(supervisor_findzero._nodeIds) )

    supervisor_findzero.connect(s)

    if(spikeStreamer != 0):
         rw_logger.info('Using Spike Streamer!')
         spikeStreamer.setupSpikeOutput(supervisor_findzero, 1000)

    ##### Connect inputs to global supervisors to kick off the network
    # We need the inputs to also kick off the buffer supervisor.  This can be tricky if the supervisor threshold is greater than the
    # maximum weight allowed in Loihi.  So we just create multiple inputs if necessary.

    threshold_wgt=threshold_super
    threshold_count=1
    if(threshold_wgt>200):
        threshold_count=math.ceil(threshold_super/200)
        threshold_wgt=math.ceil(threshold_super/threshold_count)

    total_syn_input = 0
    for i in range(0, threshold_count):
        spike_inputs.connect(supervisor, prototype=connProto, weight=np.array([[0, threshold_wgt+10], [0, 0]]), delay=np.array([1]))
        total_syn_input += threshold_wgt+10
    rw_logger.info('Setting up supervisors')
    rw_logger.info("    Total super input: {}".format( total_syn_input) ) 
    rw_logger.info("    vs threshold: {}".format( threshold_super) )
    rw_logger.info("    counter_weight: {}".format( total_counter_weight) ) 
    rw_logger.info("    counter_threshold: {}".format(1000*total_counter_weight-.5*total_counter_weight) )


    ##### Create probe for supervisors to track activity
    if track==1 or track==3:
        # Track end state
        super_counters=superNet.createCompartmentGroup(size=3, prototype=proto_counter_total)
        super_to_counters=supervisor.connect(super_counters, prototype=connProto,
                                                                        delay=np.array([1]),
                                                                        weight = np.array([[0, total_counter_weight], [0, 0], [0, 0]]),
                                                                        connectionMask=np.array([[0, 1], [0, 0], [0, 0]]))

        super_counters_place=super_counters.connect(super_counters, prototype=connProto,
                                                                        delay=np.array([1]),
                                                                        weight = np.array([[0, 0, 0], [total_counter_weight, 0, 0], [0, total_counter_weight, 0]]),
                                                                        connectionMask=np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]]))

        core_neuroncount[supervisor_core]+=3
        supervisorVProbe = super_counters.probe([nx.ProbeParameter.COMPARTMENT_VOLTAGE], customUProbeCond)
        track_probes.append([])
        track_probes.append(supervisorVProbe)
        track_probes.append([])

    if track==2 or track ==5 or track ==4 or track == 6:
        # Track all spikes
        (supervisorSProbe, supervisorVProbe, supervisorUProbe) = supervisor.probe([nx.ProbeParameter.SPIKE, nx.ProbeParameter.COMPARTMENT_VOLTAGE, nx.ProbeParameter.COMPARTMENT_CURRENT])#, [customSpikeProbeCond, customUProbeCond, customUProbeCond])
        (supervisorSProbe_zeros, supervisorVProbe_zeros, supervisorUProbe_zeros) = supervisor_findzero.probe([nx.ProbeParameter.SPIKE, nx.ProbeParameter.COMPARTMENT_VOLTAGE, nx.ProbeParameter.COMPARTMENT_CURRENT])#, [customSpikeProbeCond, customUProbeCond, customUProbeCond])
        if(track == 2 and verbose == 0):
            track_probes.append(supervisorSProbe)
            track_probes.append(['Supervisor Fire Spike', supervisorSProbe_zeros])
            track_probes.append([])
        if(track == 2 and verbose == 1):
            track_probes.append(supervisorSProbe)
            track_probes.append(supervisorVProbe)
            track_probes.append(supervisorUProbe)
            rw_logger.info('Track 2')
            rw_logger.info('    Adding Supervisor probes')
        if(track == 5):
            track_probes.append(['Supervisor Global Spike', supervisorSProbe])
            track_probes.append(['Supervisor Fire Spike', supervisorSProbe_zeros])
            track_probes.append(['Supervisor Current', supervisorUProbe])
            rw_logger.info('Track 5')
            rw_logger.info('    Adding Supervisor probes')
            rw_logger.info('        Probe 0 = main')
            rw_logger.info('        Probe 1 = Sync')
        if(track == 6):
            super_probes=[]
            super_probes.append(['Supervisor Global Spike', supervisorSProbe])
            super_probes.append(['Supervisor Fire Spike', supervisorSProbe_zeros])
            track_probes.append(super_probes)
            rw_logger.info('Track 6')
            rw_logger.info('    Adding Supervisor probes')
            rw_logger.info('        Probe 0 = main')
            rw_logger.info('        Probe 1 = Sync')

    #
    #### Create local supervisors
    #
    # Each core has a set of supervisors to monitor whether the mesh points on that core are ready to progress forward
    #     This is necessary to keep the overall core fan-in / fan-out below Loihi core limits
    #     These local supervisors communicate to the main supervisors that the core is ready to move forward.

    supervisor_local=[]
    supervisor_local_conn=[]
    supervisor_local_probes=[]

    supervisor_chip = []
    supervisor_relay = []
    supervisor2relay_conn = []
    supervisor_chip_conn = []
    supervisor_chip_probes = []
    k=0
    for i in range(0, chip_count):
        chip_start=i*128
        chip_end=(i+1)*128
        ct_chip = 0
        for j in range(chip_start, chip_end):
            if(core_meshcount[j]>0):
                ct_chip=ct_chip+1

        threshold_super_chip=weight_super*ct_chip-2
        proto_counter_super_chip=nx.CompartmentPrototype(vThMant=threshold_super_chip,
                                             logicalCoreId=i*128,
                                             compartmentVoltageDecay = 0,
                                             compartmentCurrentDecay = 4095)
        proto_counter_super_relay=nx.CompartmentPrototype(vThMant=50, logicalCoreId=i*128, compartmentVoltageDecay=0, compartmentCurrentDecay=4095)

        supervisor_chip.append(superNet.createCompartmentGroup(size = 2, prototype = proto_counter_super_chip))
        supervisor_relay.append(superNet.createCompartmentGroup(size = 2, prototype = proto_counter_super_relay))
        core_neuroncount[i*128]+=4
        supervisor_chip_conn.append(supervisor_chip[k].connect(supervisor, prototype=connProto_Ex, weight=np.array([[weight_super, 0], [0, weight_super]]), delay=np.array([1])))
        supervisor_chip_conn.append(supervisor_chip[k].connect(supervisor_chip[k], prototype=connProto_Ex, weight=np.array([[0, 4], [4, 0]]), delay=np.array([2])))


        supervisor2relay_conn.append(supervisor.connect(supervisor_relay[k], prototype=connProto_Ex, weight=np.array([[104, 0], [0, 104]]), delay=np.array([1])))
        if track==5:
              rw_logger.info('Track 5: Setting up relay on core: {} chip: {} k_ct: {} ct_chip: {} threshold_super_chip: {} weight_super: {}'.format(i*128, i, k, ct_chip, threshold_super_chip, weight_super) )
              track_probes.append(['Supervisor Relay Spike:'+ str(i) , supervisor_relay[i].probe([nx.ProbeParameter.SPIKE])])
              track_probes.append(['Supervisor Chip Spike:'+str(i), supervisor_chip[i].probe([nx.ProbeParameter.SPIKE])])

        k=k+1

    k=0
    k1=0
    for i in range(0, core_start):
        supervisor_local.append(0)
        k=k+1

    for i in range(core_start, max_coreid+1):
        # Loop through all cores with mesh points
        if(core_meshcount[i]==0):
            # If core does not have meshpoints, skip
            supervisor_local.append(0)
            k=k+1
            continue

        # Create local supervisor collectors on each core with mesh points.
        # Each supervisor neuron has a voltage threshold of 16*[# mesh points on core]
        weight_super=16
        threshold_super_local = weight_super*core_meshcount[i]-2

        proto_counter_super_local=nx.CompartmentPrototype(vThMant=threshold_super_local,
                                             logicalCoreId=i,
                                             compartmentVoltageDecay = 0,
                                             compartmentCurrentDecay = 4095)

        supervisor_local.append(superNet.createCompartmentGroup(size = 2, prototype=proto_counter_super_local))
        core_neuroncount[i]+=2

        # Local supervisors connect to global supervisors with 'weight_super' to indicate that core is ready to move forward
        supervisor_local_conn.append(supervisor_local[k].connect(supervisor_chip[math.floor(i/128)], prototype=connProto_Ex, weight=np.array([[weight_super, 0], [0, weight_super]]), delay=np.array([1])))

        supervisor_local_conn.append(supervisor_local[k].connect(supervisor_local[k], prototype=connProto_Ex, weight=np.array([[0, 4], [4, 0]]), delay=np.array([2])))


        if(track==2 or track ==5):
            # For track =2 and track =5, we want to track whether local supervisors spike
            track_probes.append(['Supervisor Local Spike :'+str(k), supervisor_local[k].probe([nx.ProbeParameter.SPIKE, nx.ProbeParameter.COMPARTMENT_VOLTAGE])])

        k=k+1
        k1=k1+1

    if(track==2 and verbose):
        supervisor_local_probes.append(supervisor_local[2].probe([nx.ProbeParameter.COMPARTMENT_VOLTAGE]))
        supervisor_local_probes.append(supervisor_local[3].probe([nx.ProbeParameter.COMPARTMENT_VOLTAGE]))

    if(track==5):
        track_probes.append(['Supervisor Core Spike :1', supervisor_local[1].probe([nx.ProbeParameter.SPIKE, nx.ProbeParameter.COMPARTMENT_VOLTAGE])])
        track_probes.append(['Supervisor Core Spike :2', supervisor_local[2].probe([nx.ProbeParameter.SPIKE, nx.ProbeParameter.COMPARTMENT_VOLTAGE])])


    rw_logger.info("Total local")
    rw_logger.info("    super input: {}".format(k1*weight_super))
    rw_logger.info("    local threshold: {}".format(threshold_super_local))
    rw_logger.info("    super_weight: {}".format(weight_super))
    rw_logger.info("    counter_threshold: {}".format(1000*total_counter_weight-.5*total_counter_weight))


    ###################################
    #### START MESH ####
    ###################################

    # Up until now, we have been setting up supervisors and mapping mesh points to cores, now we need to lay out circuit itself


    k=0
    core_id=core_start
    random_core_id=random_core_start

    # We have to keep track of stochastic neurons separately, since Loihi does not allow mixing of different stochastic types on cores
    random_neuron_count=0

    # Heuristic to keep stochastic neuron counts on cores somewhat low
    random_neuron_core_max=max(min(core_density*2, 800), 40)

    rw_logger.info('Assigned Cores')
    rw_logger.info('    Random count: {}'.format(random_neuron_core_max))
    rw_logger.info('    Core_density: {}'.format(core_density))
    counter_probes = dict()
    for i in mesh.nodes:

        k=k+1  # Count of number of mesh nodes so far for tracking purposes

        # Identify Loihi core for mesh point (per above allocation)
        core_id=mesh.nodes[i]['core_id']
        chip_id=math.floor(core_id/128)

        # Number of stochastic neurons is determined by degree of fan-out of mesh point
        prob_size=mesh.degree([i])
        print((k+0.01)/x_size/y_size, end='\r')
        # Set up prototypes for counter neurons
        total_counter_weight=2
        proto_counter_total=nx.CompartmentPrototype(vThMant=1000*total_counter_weight-.5*total_counter_weight,
                                                logicalCoreId=core_id,
                                                compartmentVoltageDecay = 0,
                                                compartmentCurrentDecay = 4095)

        # Counter neurons for mesh points are set to have threshold of 2
        proto_counter_if=nx.CompartmentPrototype(vThMant=2,
                                         logicalCoreId=core_id,
                                         compartmentVoltageDecay = 0,
                                         compartmentCurrentDecay = 4095,
                                         refractoryDelay=1)
        proto_counter_tg=nx.CompartmentPrototype(vThMant=2,
                                         logicalCoreId=core_id,
                                         compartmentVoltageDecay = 4095,
                                         compartmentCurrentDecay = 4095)


        #### Create neurons ####
        #
        # Create buffer neurons
        mesh.nodes[i]['buffer'] = superNet.createCompartmentGroup(size=3,
                                                                  prototype=[proto_counter_if, proto_counter_tg],
                                                                  prototypeMap=[0, 1, 1])
        core_neuroncount[core_id]+=3
        # Create Walker Counter and Walker Generator neurons
        mesh.nodes[i]['counters'] = superNet.createCompartmentGroup(size=3,
                                                                    prototype=[proto_counter_if, proto_counter_tg],
                                                                    prototypeMap=[0, 1, 1])
        core_neuroncount[core_id]+=3

        # Loihi has a pretty restrictive number of probes it is allowed to have on a chip.  If we want more than ~1000 mesh points
        # we need to deactivate the one probe per meshpoint.  We may be able to use activity trackers to count...

        if(k<910 and track==1):
            # Track activity of Counter neurons; track only end of runs
            # Each mesh node gets 4 trackers.  Use #2 for final count rates

            mesh.nodes[i]['total_counters']=superNet.createCompartmentGroup(size=3, prototype=proto_counter_total)
            mesh.nodes[i]['total_conn']=mesh.nodes[i]['counters'].connect(mesh.nodes[i]['total_counters'], prototype=connProto,
                                                                        delay=np.array([1]),
                                                                        weight = np.array([[-total_counter_weight, total_counter_weight, total_counter_weight], [0, 0, 0], [0, 0, 0]]),
                                                                        connectionMask=np.array([[1, 1, 1], [0, 0, 0], [0, 0, 0]]))

            mesh.nodes[i]['total_conn_place']=mesh.nodes[i]['total_counters'].connect(mesh.nodes[i]['total_counters'], prototype=connProto,
                                                                        delay=np.array([1]),
                                                                        weight = np.array([[0, 0, 0], [total_counter_weight, 0, 0], [0, total_counter_weight, 0]]),
                                                                        connectionMask=np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]]))

            track_probes.append([])
            mesh.nodes[i]['CounterVProbe'] = mesh.nodes[i]['total_counters'].probe([nx.ProbeParameter.COMPARTMENT_VOLTAGE], customUProbeCond)
            track_probes.append(mesh.nodes[i]['CounterVProbe'])
            track_probes.append([])
            track_probes.append([])
            core_neuroncount[core_id]+=3

        if(k<900 and track==2):
            # Track activity of Counter neurons; track all
            # Each mesh node gets 4 trackers.  Use #1 for counter spiking and #2 for counter voltages; comment out buffer counts for debugging

            mesh.nodes[i]['CounterSProbe'] = mesh.nodes[i]['counters'].probe([nx.ProbeParameter.SPIKE])
            track_probes.append(mesh.nodes[i]['CounterSProbe'])
            track_probes.append([])
            if(verbose==1):
                mesh.nodes[i]['BufferSProbe'] = mesh.nodes[i]['buffer'].probe([nx.ProbeParameter.SPIKE])
                track_probes.append(mesh.nodes[i]['BufferSProbe'])
            else:
                track_probes.append([])
            track_probes.append([])

        if(k<10 and track==5):
            # debug mode
            mesh.nodes[i]['CounterSProbe'] = mesh.nodes[i]['counters'].probe([nx.ProbeParameter.SPIKE])
            track_probes.append(mesh.nodes[i]['CounterSProbe'])
            mesh.nodes[i]['CounterVProbe'] = mesh.nodes[i]['counters'].probe([nx.ProbeParameter.COMPARTMENT_VOLTAGE], customUProbeCond)
            track_probes.append(mesh.nodes[i]['CounterVProbe'])
            mesh.nodes[i]['BufferSProbe'] = mesh.nodes[i]['buffer'].probe([nx.ProbeParameter.SPIKE])
            track_probes.append(mesh.nodes[i]['BufferSProbe'])
            mesh.nodes[i]['BufferVProbe'] = mesh.nodes[i]['buffer'].probe([nx.ProbeParameter.COMPARTMENT_VOLTAGE])
            track_probes.append(mesh.nodes[i]['BufferVProbe'])

        if(track==6):
            mesh.nodes[i]['CounterSProbe'] = mesh.nodes[i]['counters'].probe([nx.ProbeParameter.SPIKE])
            counter_probes[i] = ['Mesh Node: ' + str(i), mesh.nodes[i]['CounterSProbe'][0]]


        #### Create connections ###
        #
        # Connections between walker counter and walker generator
        mesh.nodes[i]['conn_walkers']=mesh.nodes[i]['counters'].connect(mesh.nodes[i]['counters'], prototype=connProto,
                                                                        delay=np.array([1]),
                                                                        weight = np.array([[-counter_weight, counter_weight, counter_weight], [-counter_weight, counter_weight, 0], [0, -counter_weight, 0]]),
                                                                        connectionMask=np.array([[1, 1, 1], [1, 1, 0], [0, 1, 0]]))

        # Connections between two buffer neurons
        mesh.nodes[i]['conn_buffer']=mesh.nodes[i]['buffer'].connect(mesh.nodes[i]['buffer'], prototype=connProto, delay=np.array([1]), weight = np.array([[-counter_weight, counter_weight, counter_weight], [-counter_weight, counter_weight, 0], [0, -counter_weight, 0]]),connectionMask=np.array([[1, 1, 1], [1, 1, 0], [0, 1, 0]]))

        # Connections between buffer neurons and walker counter
        # Note the extra delay between the buffer counter and the walker counter.  This is because we have to compensate for the connection delay to the buffer spike neuron, the neuron itself, and the connection to the walker counter

        mesh.nodes[i]['conn_buff2walk']=mesh.nodes[i]['buffer'].connect(mesh.nodes[i]['counters'],
                                                                           prototype=connProto,
                                                                           delay=np.array([[5, 5, 1], [1, 1, 1], [1, 1, 1]]),
                                                                           weight=np.array([[counter_weight, -counter_weight, -counter_weight], [0, 0, 0], [0, 0, 0]]),
                                                                           connectionMask=np.array([[1, 1, 1], [0, 0, 0], [0, 0, 0]]))

       # Connections between supervisor and other neurons
        # This is tricky.  The delays to the buffer and walker sets have to account for the relay between the counter and spiker.
        #     Similarly, the delays to the walker set have to account for the relay from the buffer neurons.  That makes the overall delay >6
        #     For this reason we add a 2nd relay supervisor neuron, which talks to the walker neurons.

        mesh.nodes[i]['super_start_buffer']=supervisor_relay[chip_id].connect(mesh.nodes[i]['buffer'], prototype=connProto, weight=np.array([[counter_weight, 0], [counter_weight, 0], [counter_weight, 0]]), delay=np.array([[1, 1],[3, 3], [5, 5]]))
        mesh.nodes[i]['super_start_counter']=supervisor_relay[chip_id].connect(mesh.nodes[i]['counters'],
                                                                prototype=connProto,
                                                                weight=np.array([[0, counter_weight], [0, counter_weight], [0, counter_weight]]),
                                                                delay=np.array([[1, 1],[3, 3], [5, 5]]))

        # Connection from buffer neurons to supervisor
        # This tells the supervisor neuron that all the walkers have been processed, and thus start the next global time step

        mesh.nodes[i]['buffer_supervisor']=mesh.nodes[i]['buffer'].connect(supervisor_local[core_id], prototype=connProto_Ex, weight=np.array([[0, 0, 0], [weight_super, 0, 0]]), delay=np.array([2]))
        mesh.nodes[i]['counter_supervisor']=mesh.nodes[i]['counters'].connect(supervisor_local[core_id], prototype=connProto_Ex, weight=np.array([[weight_super, 0, 0], [0, 0, 0]]), delay=np.array([2]))

        location = i

        # Use probCircuit script to generate stochastic neurons on current Loihi core for stochastic neurons

        (superNet, mesh, num_out) = probCircuit(superNet, mesh, location, core_id, random_core_id)

        # Track Loihi cores for stochastic neurons.  Each mesh point may have a different degree, so we need to update accordingly
        random_neuron_count+=num_out
        core_neuroncount[random_core_id]+=num_out
        if(random_neuron_count>random_neuron_core_max):
            random_core_id+=1
            random_neuron_count=num_out
            max_coreid+=1
            if(verbose):
                rw_logger.info('i: {} random_core_id: {} max_coreid: {} random_neuron_count: {} random_neuron_core_max: {}'.format(i, random_core_id, max_coreid, random_neuron_count, random_neuron_core_max) )

        core_neuroncount[core_id]+=num_out

        # Tracking: Track = 5 exports output neurons as well.
        if(track==1):
            track_probes.append([])
        if(track==2):
            track_probes.append([])
        if(track==5 and k<10):
            mesh.nodes[i]['OutputSProbe'] = mesh.nodes[i]['outputs'].probe([nx.ProbeParameter.SPIKE],customSpikeProbeCond)
            track_probes.append(mesh.nodes[i]['OutputSProbe'])

        # Create connection onto input neurons
        if(i in input_x):
            spike_inputs.connect(mesh.nodes[i]['buffer'], prototype=connProto_In, weight=np.array([[-counter_weight, 0], [0, 0], [0, 0]]), delay=np.array([3]))
    track_probes.append(counter_probes)

    rw_logger.info('Stochastic neurons setup')
    # On each stochastic core add a few randomly connected neurons to vary where we are on PRNG
    for i in range(random_core_start, random_core_id+1):
        N=2+seed_time%15
        core_neuroncount[i]+=N
        probRandomizeCircuit(superNet, i, N_random=N)

    # Loihi seems to occasionally struggle if there is a core without neurons
    for i in range(core_start, max_coreid+1):
        if(core_neuroncount[i]==0):
            proto_counter_if=nx.CompartmentPrototype(vThMant=100,
                                             logicalCoreId=i,
                                             compartmentVoltageDecay = 0,
                                             compartmentCurrentDecay = 4095,
                                             refractoryDelay=1)
            rw_logger.warning('WARNING: Empty Core: {} {} {}'.format(i, core_start, chip_max))
            superNet.createCompartmentGroup(size=1, prototype=proto_counter_if)

    if(verbose):
        for i in supervisor_local_probes:
            track_probes.append(i)

    rw_logger.info('Connecting mesh points')
    # Call 'connect_mesh' script to link mesh points together (connect outputs to buffer inputs)
    connect_mesh(mesh, counter_weight)

    # Force python to delete mesh intermediate
    del mesh
    return core_neuroncount, max_coreid, track_probes, chip_count, s
