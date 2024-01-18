# Copyright 2023 National Technology & Engineering Solutions of Sandia, LLC (NTESS). 
# Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains 
# certain rights in this software.
# Apache License Version 2.0, January 2004 http://www.apache.org/licenses/
import numpy as np
from pyNN.random import NumpyRNG, RandomDistribution
import math

simulator_Name = 'spiNNaker'
import networkx as nx


def counters_supervisor():

    w_c_s=[]
    w_g_s=[]
    d_c_s=[]
    d_g_s=[]
    conn_c_s=[]
    conn_g_s=[]

    conn2add=(0, 1)  # buffer counter -> counter supervisor
    conn_c_s.append(conn2add)
    w_c_s.append(1.0)
    d_c_s.append(3)
    conn2add=(1, 0)  # counter counter-> buffer supervisor
    conn_c_s.append(conn2add)
    w_c_s.append(1.0)
    d_c_s.append(3)

    return conn_c_s, w_c_s, d_c_s

def supervisor_counters():

    w_s_c=[]
    w_s_g=[]
    d_s_c=[]
    d_s_g=[]
    super_c=[]
    super_g=[]

    conn2add=(0, 0)  # Supervisor buffer -> counter 0
    super_c.append(conn2add)
    w_s_c.append(1.0)
    d_s_c.append(1)
    conn2add=(1, 1)  # Supervisor counter -> counter 1
    super_c.append(conn2add)
    w_s_c.append(1.0)
    d_s_c.append(1)

    conn2add=(0, 0)  # Supervisor buffer -> buffer generator 0
    super_g.append(conn2add)
    w_s_g.append(1.0)
    d_s_g.append(2)
    conn2add=(0, 1)  # Supervisor buffer -> buffer relay 1
    super_g.append(conn2add)
    w_s_g.append(1.0)
    d_s_g.append(3)
    conn2add=(1, 2)  # Supervisor counter -> counter generator 2
    super_g.append(conn2add)
    w_s_g.append(1.0)
    d_s_g.append(2)
    conn2add=(1, 3)  # Supervisor counter -> counter relay 3
    super_g.append(conn2add)
    w_s_g.append(1.0)
    d_s_g.append(3)

    return super_c, super_g, w_s_c, w_s_g, d_s_c, d_s_g

def mesh_counters():
    # Define for counters and probabilistic neurons
    w=[]
    delay=[]
    orbs_start=[]
    connection=[]

    conn_c_c=[]
    conn_c_g=[]
    conn_g_c=[]
    conn_g_g=[]

    w_c_c=[]
    d_c_c=[]
    w_c_g=[]
    d_c_g=[]
    w_g_g=[]
    d_g_g=[]
    w_g_c=[]
    d_g_c=[]

    d_default=1

    # Lay out c to c connections
    # c_0 = buffer counter
    # c_1 = counter counter

    conn2add=(0, 0)
    conn_c_c.append(conn2add)
    w_c_c.append(-1.0)
    d_c_c.append(d_default)
    conn2add=(0, 1)
    conn_c_c.append(conn2add)
    w_c_c.append(1.0)
    d_c_c.append(3.0*d_default)
    conn2add=(1, 1)
    conn_c_c.append(conn2add)
    w_c_c.append(-1.0)
    d_c_c.append(d_default)
    # Lay out c to g connections
    # c_0 = buffer counter
    # c_1 = counter counter
    # g_0 = buffer generator
    # g_1 = buffer relay
    # g_2 = counter generator
    # g_3 = counter relay

    conn2add=(0, 0)
    conn_c_g.append(conn2add)
    w_c_g.append(-1.0)
    d_c_g.append(d_default)
    conn2add=(1, 2)
    conn_c_g.append(conn2add)
    w_c_g.append(-1.0)
    d_c_g.append(d_default)

    # Lay out g to c connections
    # g_0 = buffer generator
    # g_1 = buffer relay
    # g_2 = counter generator
    # g_3 = counter relay
    # c_0 = buffer counter
    # c_1 = counter counter

    conn2add=(0, 0)  # Buffer generator -> counter
    conn_g_c.append(conn2add)
    w_g_c.append(1.0)
    d_g_c.append(d_default)
    conn2add=(1, 0)  # Buffer relay-> counter
    conn_g_c.append(conn2add)
    w_g_c.append(1.0)
    d_g_c.append(.9)
    conn2add=(2, 1)  # Counter generator -> counter
    conn_g_c.append(conn2add)
    w_g_c.append(1.0)
    d_g_c.append(d_default)
    conn2add=(3, 1)  # Counter relay -> counter
    conn_g_c.append(conn2add)
    w_g_c.append(1.0)
    d_g_c.append(d_default)
    conn2add=(0, 1)  # Counter generator -> counter
    conn_g_c.append(conn2add)
    w_g_c.append(-1.0)
    d_g_c.append(3.0*d_default)
    conn2add=(1, 1)  # Counter relay -> counter
    conn_g_c.append(conn2add)
    w_g_c.append(-1.0)
    d_g_c.append(1.0*d_default)

    # Lay out g to g connections
    # g_0 = buffer generator
    # g_1 = buffer relay
    # g_2 = counter generator
    # g_3 = counter relay

    conn2add=(0, 0)  # Buffer generator -> buffer generator
    conn_g_g.append(conn2add)
    w_g_g.append(1.0)
    d_g_g.append(d_default)
    conn2add=(0, 1)  # Buffer generator -> relay
    conn_g_g.append(conn2add)
    w_g_g.append(-1.0)
    d_g_g.append(d_default)
    conn2add=(2, 2)  # Counter generator -> buffer generator
    conn_g_g.append(conn2add)
    w_g_g.append(1.0)
    d_g_g.append(d_default)
    conn2add=(2, 3)  # Counter generator -> relay
    conn_g_g.append(conn2add)
    w_g_g.append(-1.0)
    d_g_g.append(d_default)

    return conn_c_c, conn_c_g, conn_g_c, conn_g_g, w_c_c, w_c_g, w_g_c, w_g_g, d_c_c, d_c_g, d_g_c, d_g_g

def stochastic_input(num_prob):

    conn_poisson_prob=[]
    w_poisson_prob=[]
    d_poisson_prob=[]

    # Connect Poisson inputs to prob neurons
    for i in range(0, num_prob):
        conn_poisson_prob.append((i,i))
        w_poisson_prob.append(0.5)
        d_poisson_prob.append(1.0)

    #conn_poisson_prob.append((1,1))
    #w_poisson_prob.append(0.5)
    #d_poisson_prob.append(1.0)

    #conn_poisson_prob.append((2,2))
    #w_poisson_prob.append(0.5)
    #d_poisson_prob.append(1.0)

    return conn_poisson_prob, w_poisson_prob, d_poisson_prob

def gen_prob(num_prob, num_out):
    conn_c_output=[]
    conn_c_prob=[]
    w_c_output=[]
    w_c_prob=[]
    d_c_output=[]
    d_c_prob=[]

    # Connect counter to prob neurons
    for i in range(0, num_prob):
        conn_c_prob.append((1,i))
        w_c_prob.append(-1.0)
        d_c_prob.append(1.0)

    # Connect counter to first output neuron
    conn_c_output.append((1,num_out-1))
    w_c_output.append(-1.0)
    d_c_output.append(2.0)

    return conn_c_output, conn_c_prob, w_c_output, w_c_prob, d_c_output, d_c_prob

def arb_fan(prob_weight, out_weight,num_prob, num_out):
    conn_count_output=[]
    conn_count_prob=[]
    conn_prob_output=[]
    w_count_output=[]
    w_count_prob=[]
    w_prob_output=[]
    d_count_output=[]
    d_count_prob=[]
    d_prob_output=[]
    for i in range(0, num_prob):
        conn_count_prob.append((2,i))
        w_count_prob.append(0.5)
        d_count_prob.append(2.0)

        for j in range(0, num_out):
            w = prob_weight[j][i]  #The code produces matrix in [to][from] coordinates
            if(w!=0):
                conn_prob_output.append((i,j))  #Pynn takes (from, to) coordinates
                w_prob_output.append(w)
                d_prob_output.append(1.0)

    for j in range(0, num_out):
        w=out_weight[j][0]
        if(w!=0):
            conn_count_output.append((2, j))
            w_count_output.append(1.0)
            d_count_output.append(3.0)

    return conn_count_output, conn_count_prob, conn_prob_output,  w_count_output, w_count_prob, w_prob_output, d_count_output, d_count_prob, d_prob_output

def four_fan():
    conn_count_output=[]
    conn_count_prob=[]
    conn_prob_output=[]
    w_count_output=[]
    w_count_prob=[]
    w_prob_output=[]
    d_count_output=[]
    d_count_prob=[]
    d_prob_output=[]

    # Connect counter to prob neurons
    conn_count_prob.append((2,0))
    w_count_prob.append(0.5)
    d_count_prob.append(2.0)

    conn_count_prob.append((2,1))
    w_count_prob.append(0.5)
    d_count_prob.append(2.0)

    conn_count_prob.append((2,2))
    w_count_prob.append(0.5)
    d_count_prob.append(2.0)

    # Connect counter to first output neuron
    conn_count_output.append((2,0))
    w_count_output.append(1.0)
    d_count_output.append(3.0)

    # Connect prob neurons to output neurons
    conn_prob_output.append((0,0))
    w_prob_output.append(-1.0)
    d_prob_output.append(1.0)

    conn_prob_output.append((1,0))
    w_prob_output.append(-1.0)
    d_prob_output.append(1.0)

    conn_prob_output.append((0,1))
    w_prob_output.append(1.0)
    d_prob_output.append(1.0)

    conn_prob_output.append((1,1))
    w_prob_output.append(-1.0)
    d_prob_output.append(1.0)


    # Need to fix this script to allow arbitrary fan out.  Hard coded here for 25% each direction
    #conn_prob_output.append((2,1))
    #w_prob_output.append(-1.0)
    #d_prob_output.append(1.0)

    conn_prob_output.append((0,2))
    w_prob_output.append(-1.0)
    d_prob_output.append(1.0)

    conn_prob_output.append((1,2))
    w_prob_output.append(1.0)
    d_prob_output.append(1.0)

    #conn_prob_output.append((2,2))
    #w_prob_output.append(-1.0)
    #d_prob_output.append(1.0)

    conn_prob_output.append((0,3))
    w_prob_output.append(0.5)#-1.0)
    d_prob_output.append(1.0)

    conn_prob_output.append((1,3))
    w_prob_output.append(0.5)#-1.0)
    d_prob_output.append(1.0)

    #conn_prob_output.append((2,3))
    #w_prob_output.append(1.0)
    #d_prob_output.append(1.0)

    return conn_count_output, conn_count_prob, conn_prob_output,  w_count_output, w_count_prob, w_prob_output, d_count_output, d_count_prob, d_prob_output

def normalize_synapses(w, d, w_scale, d_scale):
    w_norm=[j*w_scale for j in w]
    w=[max(j,0.0) for j in w_norm]
    w_i=[-min(j,0.0) for j in w_norm]
    d_norm=[j*d_scale for j in d]
    d=d_norm
    return w, w_i, d

def readTransitionMatrix(i, mat):
    # Read in 'i' which is current node
    # 'mat' which is the transition matrix, listed in [to, from] coordinates
    #
    # prob, i_conn = readTransitionMatrix(i, mat)
    #
    mat_i=mat[i]
    size_i, size_j = mat.shape
    i_conn=[]
    prob=[]
    for j in range(0, size_j):
        if(mat_i[j]!=0):
            i_conn.append(j)
            prob.append(mat_i[j])

    return prob, i_conn

def connect(i, grid_size):
    i_conn=[]
    #if(i+grid_size>(grid_size*grid_size)):
    #    i_n=i-grid_size
    #else:
    #    i_n=i+grid_size
    #if(i-grid_size<0):
    #    i_s=i+grid_size*grid_size-grid_size
    #else:
    #    i_s=i-grid_size
    #i_conn.append(i_n)
    #i_conn.append(i_s)
    i_conn.append((i+grid_size)%(grid_size*grid_size))
    i_conn.append((i-grid_size)%(grid_size*grid_size))
    if(i%grid_size==0):
        #i_w=i+1
        i_w=i+grid_size-1
    else:
        i_w=i-1
    if((i+1)%grid_size==0):
        i_e=i-grid_size+1
        #i_e=i-1
    else:
        i_e=i+1
    i_conn.append(i_w%(grid_size*grid_size))
    i_conn.append(i_e%(grid_size*grid_size))
    return i_conn

def connect_synapses(k):
    conn_i1_i2=[(k,0)]
    w_i1_i2=[(-1.0)]
    d_i1_i2=[(1.0)]
    return conn_i1_i2, w_i1_i2, d_i1_i2



def get_probCircuitValues(prob):
    #####
    # Adapted code from Loihi
    #
    # Script to compute probability values for arbitrary transition matrix fanouts
    #
    # This script implements Algorithm 1 in Additional Material

    # We compute probabilities on a base-2 tree, so add probabilities up to nearest power of 2

    # Fill in probabilities to we get an effective tree that is a power of 2; keep track of 0 probabilities
    #     For example, 5 outputs would become 8, with the final 3 probabilities with p=0
    no_prob_count=0
    while(len(prob)<2**math.ceil(math.log(len(prob),2))):
        prob.append(0)
        no_prob_count+=1

    # Convert probability list to numpy vector.  Normalize to ensure all probabilities equal 1
    #        Note: this normalization many not be appropriate for all applications
    #
    prob=np.asarray(prob)
    if(np.sum(prob)>1):
        prob=prob/np.sum(prob)
    elif(np.sum(prob)<1 and prob[0]==0):
        prob[0]=1-np.sum(prob)
    elif(np.sum(prob)<1 and prob[0]!=0):
        prob=prob/np.sum(prob)


    # Create probability tree tracking vectors
    p=np.zeros(prob.shape)
    total_branch_prob=np.zeros(prob.shape)
    delete_neuron=[]
    conn = np.zeros((len(prob), len(prob)))
    k=0
    layer=0
    k_old=[]
    k_old.append(k)
    conn_targets=[]

    # Compute first layer of tree probabilities based on provided probs.  Each iteration represents a probability neuron
    for i in range(0, int(len(p)/2)):

        # Compute probability that this neuron should fire
        if((prob[i*2+1]+prob[i*2])>0):
            p[i]=prob[i*2]/(prob[i*2+1]+prob[i*2])

        # Track cumulative probability that this neuron should influence
        total_branch_prob[i]=prob[i*2+1]+prob[i*2]

        # Identify which output neurons that this probability neuron impacts (positive and negative)
        conn[i*2, i+1]=1
        conn[i*2+1, i+1]=-1

        # Track targets for probability neuron (we will use this for higher up tree)
        conn_targets.append([i*2, i*2+1])
        k+=1

        # Delete check identifies outputs without any probability and flags probability neurons that don't need to exist
        delete_check=1
        if(prob[i*2]>0):
            delete_check=0
        if(prob[i*2+1]>0):
            delete_check=0
        delete_neuron.append(delete_check)

    # We're now going to walk up the 'tree' and use the probabilities of previous layer
    layer_max=k+int((k-k_old[layer])/2)
    layer_min=k
    layer=1
    k_old.append(k)
    while(k<(len(prob)-1)):
        # We will keep adding layers to the tree until we are out of probability neurons
        layer_max=k+int((k-k_old[layer-1])/2)
        layer_min=k
        for i in range(layer_min, layer_max):
            index_1=(i-layer_min)*2+k_old[layer-1]
            index_2=(i-layer_min)*2+k_old[layer-1]+1

            # Determine total output probability this prob neuron accounts for based on downstream branches
            total_branch_prob[i]=(total_branch_prob[index_1]+total_branch_prob[index_2])
            total=(total_branch_prob[index_1]+total_branch_prob[index_2])

            # Compute probability that this neuron should fire based on downstream branch probabilities
            if(total>0):
                p[i]=total_branch_prob[index_1]/total

            # We identify which neurons that this probability neuron contributes to and do our delete check
            conn_targets_here=[]
            delete_check=1
            for j in conn_targets[(i-layer_min)*2+k_old[layer-1]]:
                conn[j, i+1]=1
                conn_targets_here.append(j)
                if(prob[j]>0):
                    delete_check=0
            for j in conn_targets[(i-layer_min)*2+1+k_old[layer-1]]:
                conn[j, i+1]=-1
                conn_targets_here.append(j)
                if(prob[j]>0):
                    delete_check=0
            delete_neuron.append(delete_check)
            conn_targets.append(conn_targets_here)

            k+=1
        k_old.append(k)
        layer+=1



    # Delete probability neurons that have no outputs

    for i in reversed(range(0, len(delete_neuron))):
        if(delete_neuron[i]==1):
            conn=np.delete(conn, i+1, 1)
            p=np.delete(p, i)

    # Delete output neurons that have no probability of firing

    for i in reversed(range(0, len(prob))):
        if(prob[i]==0):
            conn=np.delete(conn, i, 0)

    num_prob, num_out = conn.shape

    # Delete probability neurons for outputs that really just rely on all inputs being off

    for i in reversed(range(1, num_out)):
        if(sum(conn[conn[:,i]<0,i])==0):
            conn=np.delete(conn, i, 1)
            p=np.delete(p, i-1)

    num_prob, num_out = conn.shape
    # Normalize Conn matrix to synaptic weights to ensure that all positive inputs fire and one negative input can suppress

    for i in range(0, len(conn)):
        conn[i,conn[i,:]==1]=1.20/max(sum(conn[i,:]==1),1)
        conn[i,conn[i,:]==-1]=-1.00

    relay2out_weight=np.zeros((num_out, 1))
    for i in range(0, num_out):
        if(np.max(conn[i,:])<=0):
            relay2out_weight[i]=1.20

    weight=np.zeros((num_prob,1))
    for i in range(1, num_prob):
        #weight[i]=128*p[i-1]+36
        weight[i]=p[i-1]


    return conn, relay2out_weight, weight, p

def setup_network(pynn,
                  num_time,
                  num_walkers,
                  starting_node,
                  matrix_filename='curmat.csv',
                  seed = 1):
    #### Set up mesh

    D=1
    sim_time = num_time

    mat = np.genfromtxt(matrix_filename, delimiter=',')
    #print(mat)
    n_meshpoints, tmp = mat.shape

    pynn.setup(timestep = 1.0, min_delay = 1.0, max_delay = 32.0)
    if simulator_Name == "spiNNaker":
        #pynn.set_number_of_neurons_per_core(pynn.IF_curr_exp, 250)
        pynn.set_number_of_neurons_per_core(pynn.extra_models.IF1_curr_delta, 50)
        pynn.set_number_of_neurons_per_core(pynn.extra_models.IF0_curr_delta, 175)

    rng = NumpyRNG(seed=seed)

    #tau_syn=tauSyn

    # Define SpiNNaker neuron types (TG and IF)
    tg_params = {'cm': 1000, 'tau_m': 1000, 'v_rest': 0.0, 'v_thresh':1, 'i_offset':0.0}
    if_params = {'cm': 1000, 'tau_m': 1000,'v_rest': 0.0, 'v_thresh':1, 'i_offset':0.0}
    if_params_supervisor = {'cm': 1000, 'tau_m': 1000,'v_rest': 0.0, 'v_thresh':n_meshpoints+.25, 'i_offset':0.0}

    #Spike_input = pynn.Population(1, pynn.SpikeSourceArray([1, 2, 3, 4, 5, 6]), label="Input")

    #pop_supervisors=pynn.Population(2, pynn.IF_curr_exp(**if_params), label="supervisors")
    pop_supervisors=pynn.Population(2, pynn.extra_models.IF1_curr_delta(**if_params_supervisor), label="supervisors")

    pop_supervisors.initialize(v=[n_meshpoints+2, 0])
    pop_supervisors.record(["spikes", "v"])

    conn_s_s, w_s_s, d_s_s = counters_supervisor()
    walker_conn_super_super=pynn.FromListConnector(conn_s_s)
    conn_s_s=pynn.Projection(pop_supervisors, pop_supervisors, walker_conn_super_super, receptor_type='excitatory', synapse_type=pynn.StaticSynapse(weight=w_s_s, delay=d_s_s))


    pop_counter=[]
    pop_generators=[]
    pop_prob=[]
    pop_outputs=[]

    for i in range(0, n_meshpoints):
        w_scale=1.0
        d_scale=1.0
        label_pop= "walker " +'i'
        # Create neurons
        #pynn.IF_curr_exp(**if_params) --> extra_models.IF1_curr_delta
        pop_counter.append(pynn.Population(2, pynn.extra_models.IF1_curr_delta(**if_params), label="counters"))
        pop_generators.append(pynn.Population(4, pynn.extra_models.IF0_curr_delta(**tg_params), label="counters_spike"))

        # Initialize neurons
        pop_counter[i].initialize(v=[0, 0])
        pop_generators[i].initialize(v=[0, 0, 0, 0])

        conn_c_c, conn_c_g, conn_g_c, conn_g_g, w_c_c, w_c_g, w_g_c, w_g_g, d_c_c, d_c_g, d_g_c, d_g_g = mesh_counters()

        walker_conn_c_c=pynn.FromListConnector(conn_c_c)
        walker_conn_c_g=pynn.FromListConnector(conn_c_g)
        walker_conn_g_c=pynn.FromListConnector(conn_g_c)
        walker_conn_g_g=pynn.FromListConnector(conn_g_g)
        walker_conn_c_c_i=pynn.FromListConnector(conn_c_c)
        walker_conn_c_g_i=pynn.FromListConnector(conn_c_g)
        walker_conn_g_c_i=pynn.FromListConnector(conn_g_c)
        walker_conn_g_g_i=pynn.FromListConnector(conn_g_g)

        w_c_c, w_c_c_i, d_c_c = normalize_synapses(w_c_c, d_c_c, w_scale, d_scale)
        w_c_g, w_c_g_i, d_c_g = normalize_synapses(w_c_g, d_c_g, w_scale, d_scale)
        w_g_c, w_g_c_i, d_g_c = normalize_synapses(w_g_c, d_g_c, w_scale, d_scale)
        w_g_g, w_g_g_i, d_g_g = normalize_synapses(w_g_g, d_g_g, w_scale, d_scale)

        # Compute transition probabilities and stochastic neurons
        prob, i_conn = readTransitionMatrix(i, mat)
        if(len(prob)!=0):
            [conn, relay2out, p, p_w]=get_probCircuitValues(prob)
            num_out, num_prob = conn.shape
        else:
            num_prob = 0
            num_out = 0
            conn=[]
            relay2out=[]
            p=[]
        # Conn = weight matrix between probabilistic neurons and output neurons
        # relay2Out = weight matrix between generator to output neurons
        # p = stochastic neuron probabilities
        # num_prob = number of probabilistic neurons
        # num_out = number of output neurons

        # Create random neurons
        if(num_prob>0):
            Poiss_ext_E = pynn.Population(num_prob, pynn.SpikeSourcePoisson(rate=[int(1000*i1) for i1 in p]), label="Poisson_pop_E")

            # Create probabilistic neurons
            pop_prob.append(pynn.Population(num_prob, pynn.extra_models.IF0_curr_delta(**tg_params), label="prob_neurons"))
            pop_outputs.append(pynn.Population(num_out, pynn.extra_models.IF0_curr_delta(**tg_params), label="output_neurons"))
            pop_prob[i].initialize(v=0.0)#[0]*num_prob) # does this need to be a list of zeros of length num_prob?
            pop_outputs[i].initialize(v=0.0)#[0]*num_out) # does this need to be a list of zeros of length num_out?

            conn_poisson_prob, w_poisson_prob, d_poisson_prob = stochastic_input(num_prob)
            walker_conn_poisson_prob = pynn.FromListConnector(conn_poisson_prob)
            w_poisson_prob, tmp, d_poisson_prob = normalize_synapses(w_poisson_prob, d_poisson_prob, w_scale, d_scale)


            conn_c_output, conn_c_prob, w_c_output, w_c_prob, d_c_output, d_c_prob = gen_prob(num_prob, num_out)
            conn_g_output, conn_g_prob, conn_prob_output, w_g_output, w_g_prob, w_prob_output, d_g_output, d_g_prob, d_prob_output = four_fan()
            conn_g_output, conn_g_prob, conn_prob_output, w_g_output, w_g_prob, w_prob_output, d_g_output, d_g_prob, d_prob_output = arb_fan(conn, relay2out, num_prob, num_out)

            walker_conn_g_output = pynn.FromListConnector(conn_g_output)
            walker_conn_g_prob = pynn.FromListConnector(conn_g_prob)
            walker_conn_c_output = pynn.FromListConnector(conn_c_output)
            walker_conn_c_prob = pynn.FromListConnector(conn_c_prob)

            if(num_prob!=1):
                walker_conn_prob_output = pynn.FromListConnector(conn_prob_output)
                walker_conn_prob_output_i = pynn.FromListConnector(conn_prob_output)

            w_g_output, w_g_output_i, d_g_output = normalize_synapses(w_g_output, d_g_output, w_scale, d_scale)
            w_g_prob, w_g_prob_i, d_g_prob = normalize_synapses(w_g_prob, d_g_prob, w_scale, d_scale)
            w_c_output, w_c_output_i, d_c_output = normalize_synapses(w_c_output, d_c_output, w_scale, d_scale)
            w_c_prob, w_c_prob_i, d_c_prob = normalize_synapses(w_c_prob, d_c_prob, w_scale, d_scale)
            w_prob_output, w_prob_output_i, d_prob_output = normalize_synapses(w_prob_output, d_prob_output, w_scale, d_scale)

        # Create Supervisor Counters
        super_c, super_g, w_s_c, w_s_g, d_s_c, d_s_g= supervisor_counters()
        walker_conn_super_c=pynn.FromListConnector(super_c)
        walker_conn_super_g=pynn.FromListConnector(super_g)

        w_s_c, w_s_c_i, d_s_c = normalize_synapses(w_s_c, d_s_c, w_scale, d_scale)
        w_s_g, w_s_g_i, d_s_g = normalize_synapses(w_s_g, d_s_g, w_scale, d_scale)

        conn_c_s, w_c_s, d_c_s = counters_supervisor()
        walker_conn_c_super=pynn.FromListConnector(conn_c_s)

        w_c_s, w_c_s_i, d_c_s = normalize_synapses(w_c_s, d_c_s, w_scale, d_scale*2)

        pynn.Projection(pop_counter[i], pop_counter[i], walker_conn_c_c, receptor_type='excitatory', synapse_type=pynn.StaticSynapse(weight=w_c_c, delay=d_c_c))
        pynn.Projection(pop_counter[i], pop_counter[i], walker_conn_c_c_i, receptor_type='inhibitory', synapse_type=pynn.StaticSynapse(weight=w_c_c_i, delay=d_c_c))

        pynn.Projection(pop_counter[i], pop_generators[i], walker_conn_c_g, receptor_type='excitatory', synapse_type=pynn.StaticSynapse(weight=w_c_g, delay=d_c_g))
        pynn.Projection(pop_counter[i], pop_generators[i], walker_conn_c_g_i, receptor_type='inhibitory', synapse_type=pynn.StaticSynapse(weight=w_c_g_i, delay=d_c_g))

        pynn.Projection(pop_generators[i], pop_counter[i], walker_conn_g_c, receptor_type='excitatory', synapse_type=pynn.StaticSynapse(weight=w_g_c, delay=d_g_c))
        pynn.Projection(pop_generators[i], pop_counter[i], walker_conn_g_c_i, receptor_type='inhibitory', synapse_type=pynn.StaticSynapse(weight=w_g_c_i, delay=d_g_c))

        pynn.Projection(pop_generators[i], pop_generators[i], walker_conn_g_g, receptor_type='excitatory', synapse_type=pynn.StaticSynapse(weight=w_g_g, delay=d_g_g))
        pynn.Projection(pop_generators[i], pop_generators[i], walker_conn_g_g_i, receptor_type='inhibitory', synapse_type=pynn.StaticSynapse(weight=w_g_g_i, delay=d_g_g))

        pynn.Projection(pop_supervisors, pop_counter[i], walker_conn_super_c, receptor_type='excitatory', synapse_type=pynn.StaticSynapse(weight=w_s_c, delay=d_s_c))
        pynn.Projection(pop_supervisors, pop_generators[i], walker_conn_super_g, receptor_type='excitatory', synapse_type=pynn.StaticSynapse(weight=w_s_g, delay=d_s_g))

        pynn.Projection(pop_counter[i], pop_supervisors, walker_conn_c_super, receptor_type='excitatory', synapse_type=pynn.StaticSynapse(weight=w_c_s, delay=d_c_s))
        if(num_prob>0):
            if(num_prob!=1):
                pynn.Projection(pop_prob[i], pop_outputs[i], walker_conn_prob_output, receptor_type='excitatory', synapse_type=pynn.StaticSynapse(weight=w_prob_output, delay=d_prob_output))
                pynn.Projection(pop_prob[i], pop_outputs[i], walker_conn_prob_output_i, receptor_type='inhibitory', synapse_type=pynn.StaticSynapse(weight=w_prob_output_i, delay=d_prob_output))

            pynn.Projection(pop_counter[i], pop_prob[i], walker_conn_c_prob, receptor_type='inhibitory', synapse_type=pynn.StaticSynapse(weight=w_c_prob_i, delay=d_c_prob))
            pynn.Projection(pop_counter[i], pop_outputs[i], walker_conn_c_output, receptor_type='inhibitory', synapse_type=pynn.StaticSynapse(weight=w_c_output_i, delay=d_c_output))

            pynn.Projection(Poiss_ext_E, pop_prob[i], walker_conn_poisson_prob, receptor_type = 'excitatory', synapse_type=pynn.StaticSynapse(weight=w_poisson_prob, delay=d_poisson_prob))

            pynn.Projection(pop_generators[i], pop_prob[i], walker_conn_g_prob, receptor_type='excitatory', synapse_type=pynn.StaticSynapse(weight=w_g_prob, delay=d_g_prob))
            pynn.Projection(pop_generators[i], pop_outputs[i], walker_conn_g_output, receptor_type='excitatory', synapse_type=pynn.StaticSynapse(weight=w_g_output, delay=d_g_output))


        # Set up neuron tracking
        if(n_meshpoints < 10):
            pop_counter[i].record(["spikes", "v"])
            pop_generators[i].record(["spikes", "v"])
            if(num_prob>0):
                pop_prob[i].record(["spikes", "v"])
                pop_outputs[i].record(["spikes", "v"])
        else:
            pop_generators[i].record(["spikes"])

    # Connect mesh points
    for i in range(0, n_meshpoints):
        prob, i_conn = readTransitionMatrix(i, mat)
        # i_conn=connect(i, int(n_meshpoints**0.5))
        k=0
        for j in i_conn:
            # print(i, j, k)
            if(j<n_meshpoints):
                conn_i1_i2, w_i1_i2, d_i1_i2 = connect_synapses(k)
                walker_conn_i1_i2 = pynn.FromListConnector(conn_i1_i2)
                tmp, w_i1_i2, d_i1_i2 = normalize_synapses(w_i1_i2, d_i1_i2, w_scale, d_scale)
                conn_i1_i2 = pynn.Projection(pop_outputs[i], pop_counter[j], walker_conn_i1_i2, receptor_type='inhibitory', synapse_type=pynn.StaticSynapse(weight=w_i1_i2, delay=d_i1_i2))
                k=k+1

    pop_counter[starting_node].initialize(v=[-num_walkers, 0])
    pynn.run(sim_time)
    return pop_generators, pop_supervisors

def analyze_rw(pop_generators, pop_supervisors, n_meshpoints, sim_time):

    output=[]
    output_np=np.zeros((n_meshpoints, int(sim_time)))
    mesh_id=0
    for mesh_point in pop_generators:
        mesh_segs = mesh_point.get_data("spikes")
        segs=mesh_segs.segments[0]
        k=0
        for spiketrain in segs.spiketrains:
            if(k==2):
                # k=2 for the final counter
                y=spiketrain
                output.append(y)
                for i in spiketrain.times:
                    output_np[mesh_id, int(i)]=1
            k=k+1
        mesh_id+=1

    a=pop_supervisors.get_data("spikes")
    spiketrain=a.segments[0]
    super_np=np.zeros((1, int(sim_time)))
    ct_bins=0

    for i in spiketrain.spiketrains[0].times:
        super_np[0, int(i)]=1
        ct_bins = ct_bins+1

    print(ct_bins)
    output_bins = np.zeros((n_meshpoints, ct_bins))
    for m in range(0, n_meshpoints):
        i_old=0
        bin_id=0
        for i in spiketrain.spiketrains[0].times:
            bin_sum = np.sum(output_np[m,i_old:int(i)], axis=0)
            if(bin_sum>0):
                output_bins[m,bin_id]=bin_sum - 1


            i_old=int(i)
            bin_id+=1

    return output_bins, ct_bins
