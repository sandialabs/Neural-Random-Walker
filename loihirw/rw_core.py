# Copyright 2023 National Technology & Engineering Solutions of Sandia, LLC (NTESS). 
# Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains 
# certain rights in this software.
# Apache License Version 2.0, January 2004 http://www.apache.org/licenses/
import math
import numpy as np
import nxsdk.api.n2a as nx
import networkx as networkx
from nxsdk.graph.monitor.probes import *

def get_probCircuitValues(prob):
    #####
    # Script to compute probability values for arbitrary transition matrix fanouts
     
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
        conn[i,conn[i,:]==1]=120/max(sum(conn[i,:]==1),1)
        conn[i,conn[i,:]==-1]=-100    
    
    relay2out_weight=np.zeros((num_out, 1))
    for i in range(0, num_out):
        if(np.max(conn[i,:])<=0):
            relay2out_weight[i]=120

    weight=np.zeros((num_prob,1))
    for i in range(1, num_prob):
        weight[i]=128*p[i-1]+36
    

    return conn, relay2out_weight, weight
    

def probCircuit(net, mesh, location, core_reg, core_prob):
    ##### 
    # Script to generate stochastic probability neuron circuit weighted by mesh transition probabilities
    #
    # For each mesh point, we are going to generate stochastic neurons on core_prob and output neurons on core_reg
    

    # Potentially we can expand precision range of probabilities by using more stochastic neurons; we don't do that yet...
    weight_precision_factor=1
    
    proto_out=nx.CompartmentPrototype(vThMant=100, 
                          logicalCoreId=core_reg,
                          compartmentVoltageDecay = 4095, 
                          compartmentCurrentDecay=4095) 
    proto_prob=nx.CompartmentPrototype(vThMant=weight_precision_factor*100,  
                          compartmentVoltageDecay = 4095, 
                          compartmentCurrentDecay = 4095, 
                          enableNoise=1, 
                          logicalCoreId=core_prob,
                          functionalState=nx.COMPARTMENT_FUNCTIONAL_STATE.IDLE,
                          randomizeVoltage=1, 
                          noiseMantAtCompartment=0, 
                          noiseExpAtCompartment=12+math.log(weight_precision_factor,2))
    
    connProto = nx.ConnectionPrototype(signMode=nx.SYNAPSE_SIGN_MODE.MIXED)
    connProto_Ex = nx.ConnectionPrototype(compressionMode=3, numWeightBits=8, signMode=nx.SYNAPSE_SIGN_MODE.EXCITATORY)
    connProto_In = nx.ConnectionPrototype(signMode=nx.SYNAPSE_SIGN_MODE.INHIBITORY)
    
    # Identify fan-out of mesh point
    i=location
    num_outputs=len(mesh.adj[i])
    
    if(num_outputs==0):
        # If mesh point does not have outputs, we generate dummy circuit and move on
        mesh.nodes[i]['probability_circuit'] = net.createCompartmentGroup(size=4, prototype=proto_prob)
        mesh.nodes[i]['outputs'] = net.createCompartmentGroup(size=4, prototype=proto_out)
        num_out=0
        return net, mesh, num_out
    
    # We are going to build a surrogate probability tree to give us all our required transition matrix probabilities
    tree_size=math.ceil(np.log2(num_outputs))
    
    if(tree_size<=2):
        # For 4 or fewer outputs we have a hard-coded probability circuit
        prob_1 = 0
        prob_2 = 0
        prob_3 = 0
        prob_4 = 0
        weights=mesh.adj[i]
        weights_simple=[]
        for output in mesh.adj[i].keys():
            if('weight' in weights[output]):
                weights_simple.append([weights[output]['weight'], output])
            else:
                weights_simple.append([.25, output])
        
        weight_ids= list(sorted(weights_simple))
        
        if(num_outputs>1):
            prob_2=weight_ids[1][0]
            if(num_outputs>2):
                prob_3=weight_ids[2][0]
                if(num_outputs>3):
                    prob_4=weight_ids[3][0]

        prob_1=1-prob_2-prob_3-prob_4
        
        p2=prob_1/(prob_1+prob_2)
        if(num_outputs>2):
            p3=prob_3/(prob_3+prob_4)
        else:
            p3=1.0
        p1=(prob_1+prob_2)

        weight1=128.0*p1+36
        weight2=128.0*p2+36
        weight3=128.0*p3+36

        # Connections between walker counter and generator and probability gate
        
        weight_target_1=math.ceil(weight_precision_factor*weight1)
        weight_target_2=math.ceil(weight_precision_factor*weight2)
        weight_target_3=math.ceil(weight_precision_factor*weight3)
        weight_add_1=math.ceil(weight_target_1/weight_precision_factor)
        weight_add_2=math.ceil(weight_target_2/weight_precision_factor)
        weight_add_3=math.ceil(weight_target_3/weight_precision_factor)


        mesh.nodes[i]['probability_circuit'] = net.createCompartmentGroup(size=4, prototype=proto_prob)

        # Create Output Gates
        mesh.nodes[i]['outputs'] = net.createCompartmentGroup(size=4, prototype=proto_out)

        mesh.nodes[i]['conn_walk2output']=mesh.nodes[i]['counters'].connect(mesh.nodes[i]['outputs'],
                                                                     prototype=connProto,
                                                                     delay=np.array([3]),
                                                                     weight=np.array([[-30, 0, 0], [-30, 0, 0], [-30, 0, 0], [-30, 120, 0]])) 


        for j in range(0, weight_precision_factor-1):
            mesh.nodes[i]['conn_walk2prob']=mesh.nodes[i]['counters'].connect(mesh.nodes[i]['probability_circuit'],
                                                                          prototype=connProto,
                                                                          weight=np.array([[0, 0, 0], [-100, weight_add_1, 0], [-100, weight_add_2, 0], [-100, weight_add_3, 0]]), 
                                                                          delay=np.array([1]),
                                                                          connectionMask=np.array([[0, 0, 0], [1, 1, 0], [1, 1, 0], [1, 1, 0]]))
            weight_target_1-=weight_add_1
            weight_target_2-=weight_add_2
            weight_target_3-=weight_add_3
        
            
        mesh.nodes[i]['conn_walk2prob']=mesh.nodes[i]['counters'].connect(mesh.nodes[i]['probability_circuit'],
                                                                          prototype=connProto,
                                                                          weight=np.array([[0, 0, 0], [-100, weight_target_1, 0], [-100, weight_target_2, 0], [-100, weight_target_3, 0]]), 
                                                                          delay=np.array([1]),
                                                                          connectionMask=np.array([[0, 0, 0], [1, 1, 0], [1, 1, 0], [1, 1, 0]]))
            
        # Connections between probability gates and output gates
        mesh.nodes[i]['conn_prob2output']=mesh.nodes[i]['probability_circuit'].connect(mesh.nodes[i]['outputs'],
                                                                      prototype=connProto,
                                                                      weight=np.array([[0, 60, 60, 0], [0, 120, -60, 0],[0, -60, 0, 120],[0, -60, 0, -60]]), 
                                                                      delay=np.array([1]))
        num_out=4
    
    else:
        ##
        # For greater than 4 outputs, we have a more complex process to determine the probability circuit
        
        
        weights=mesh.adj[i]
        weights_simple=[]
        for output in mesh.adj[i].keys():
            if('weight' in weights[output]):
                weights_simple.append([weights[output]['weight'], output])
            else:
                weights_simple.append([.25, output])
        
        weights_ids= list(sorted(weights_simple))
        prob=[]
        #print(i, weights_ids)
        for j in weights_ids:
            prob.append(j[0])

        # Call get_probCircuitValues to determine connections and weights appropriate for this depth probability tree
        conn, relay2out, prob_weight = get_probCircuitValues(prob)

        
        num_out, num_prob = conn.shape

        
        # Set up direct connections from counter to outputs
        walk2output_weights=np.concatenate((-30*np.ones((num_out,1)), relay2out, 0*np.ones((num_out,1))), axis = 1)
        
        # Set up connections from stochastic neurons to outputs
        walk2prob_weights=np.concatenate((-100*np.ones((num_out,1)), prob_weight, 0*np.ones((num_out,1))), axis = 1)
        

        # Create Probability neurons
        mesh.nodes[i]['probability_circuit'] = net.createCompartmentGroup(size=num_prob, prototype=proto_prob)

        # Create Output Neurons
        mesh.nodes[i]['outputs'] = net.createCompartmentGroup(size=num_out, prototype=proto_out)

        mesh.nodes[i]['conn_walk2output']=mesh.nodes[i]['counters'].connect(mesh.nodes[i]['outputs'],
                                                                     prototype=connProto,
                                                                     delay=np.array([3]),
                                                                     weight=walk2output_weights)

        mesh.nodes[i]['conn_walk2prob']=mesh.nodes[i]['counters'].connect(mesh.nodes[i]['probability_circuit'],
                                                                          prototype=connProto,
                                                                          weight=walk2prob_weights, 
                                                                          delay=np.array([1]),
                                                                          connectionMask=(walk2prob_weights!=0))
            
        # Connections between probability gates and output gates
        mesh.nodes[i]['conn_prob2output']=mesh.nodes[i]['probability_circuit'].connect(mesh.nodes[i]['outputs'],
                                                                      prototype=connProto,
                                                                      weight=conn, 
                                                                      delay=np.array([1]))
   
            
    return net, mesh, num_out

