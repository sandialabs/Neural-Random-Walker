# Copyright 2023 National Technology & Engineering Solutions of Sandia, LLC (NTESS). 
# Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains 
# certain rights in this software.
# Apache License Version 2.0, January 2004 http://www.apache.org/licenses/
import sys
import numpy as np
import networkx as networkx
from nxsdk.graph.monitor.probes import *
import logging

def process_outputs_ends(track_probes, probe_start=4, probe_stride=5, num_meshpoints = 41, num_runs = 1):
    ####
    # Script to process data from probes.  This is based on Track = 1 or Track = 3 simulations, where mesh activity is accumulated
    #    in counter neurons, and the final voltage of those neurons is equivalent to the cumulative density of neurons at those locations
    #
    
    count=np.zeros((num_meshpoints, num_runs))
    num_steps=np.zeros((1, num_runs))
    mesh_node=4

    if(num_runs == 1):
        num_steps[0, 0]=track_probes[1][0].data[:][0][0]+1000*track_probes[1][0].data[:][1][0]+1000*1000*track_probes[1][0].data[:][2][0]
        num_steps[0, 0]=num_steps[0,0]/128.
        for j in range(0, num_meshpoints):
            mesh_node=probe_start+j*probe_stride
            count[j]=track_probes[mesh_node][0].data[:][0][0]+1000*track_probes[mesh_node][0].data[:][1][0]+1000*1000*track_probes[mesh_node][0].data[:][2][0]
    
    else:
        for k in range(0, num_runs):
            num_steps[0,k]=track_probes[k][1][0].data[:][0][0]+1000*track_probes[k][1][0].data[:][1][0]+1000*1000*track_probes[k][1][0].data[:][2][0]
            num_steps[0,k]=num_steps[0,k]/128.

            for j in range(0, num_meshpoints):
                mesh_node=probe_start+j*probe_stride
                count[j,k]=track_probes[k][mesh_node][0].data[:][0][0]+1000*track_probes[k][mesh_node][0].data[:][1][0]+1000*1000*track_probes[k][mesh_node][0].data[:][2][0]
    return count, num_steps

def count_steps(track_probes, probe_start=4, probe_stride=5, num_meshpoints = 41, num_runs = 1):
    ####
    # Simple script to look at supervisor counts to determine the number of time steps simulation ran
    # Based on Track = 1 or Track = 3 simulation
    
    for k in range(0, num_runs):
        if(track_probes[k]):
            if(isinstance(track_probes[k][0], str)):
                track_probes[k]=track_probes[k][1]
    num_steps=np.zeros((num_runs,1))
    if(num_runs == 1):
        num_steps[0]=track_probes[1][0].data[:][0][0]+1000*track_probes[1][0].data[:][1][0]+1000*1000*track_probes[1][0].data[:][2][0]
        num_steps[0]=num_steps[0,0]/128.
    else:
        for k in range(0, num_runs):
            num_steps[k]=track_probes[k][1][0].data[:][0][0]+1000*track_probes[k][1][0].data[:][1][0]+1000*1000*track_probes[k][1][0].data[:][2][0]
            num_steps[k]=num_steps[k]/128.
    return num_steps

def process_outputs(track_probes, probe_start=3, probe_stride=5, num_meshpoints = 41):
    #####
    # Script to convert probe tracks into a density over time for each tracked mesh point
    # This applies to Track = 2 or Track = 5
    #
    # Probes are typically ordered as follows
    #
    # 0-2 Supervisor probes
    # 3-7 Mesh point '0'
    # 3+5*i - 7+5*i -- Mesh point 'i'
    # 
    # Probe start is set to vary according to number of pre-mesh node probes
    # Probe stride is set in case there are more than 5 probes per mesh point
    
    # Determine the end times of each model update step
    #     This is because model updates are non-deterministic length in simulation timesteps

    for k in range(0, len(track_probes)):
        if(track_probes[k][0]):
            if(isinstance(track_probes[k][0], str)):
                track_probes[k]=track_probes[k][1]
    bins=np.nonzero(track_probes[probe_start][0].data[:][0])
    num_bins=np.size(bins)
    count=np.zeros((num_meshpoints, num_bins+1))
    if(num_meshpoints>0):
        mesh_node=3
        for j in range(0, num_meshpoints):
            
            mesh_node=probe_start+j*probe_stride

            spike_array=np.asarray(track_probes[mesh_node][0].data[:][0])
            track_nz=np.nonzero(track_probes[mesh_node][0].data[:][0])
            bins=track_nz[0]
            bin_start=0
            num_bins=np.size(track_nz)
            track_j=track_probes[mesh_node][0].data[:][1]
            for i in range(0, num_bins):
                bin_end=bins[i]
                count[j,i]=np.sum(track_j[bin_start:bin_end])
                bin_start=bin_end+1
    else:
        track_nz=np.nonzero(track_probes[0][0].data[:][0])
        num_bins=np.size(track_nz)
        count=0
    return count, num_bins

def process_outputs_verbose(track_probes, probe_start=3, probe_stride=5, num_meshpoints = 41):

    bins=np.nonzero(track_probes[probe_start][0].data[:][0])

    num_bins=np.size(bins)
    count=np.zeros((num_meshpoints, num_bins+1))

    mesh_node=3
    for j in range(0, num_meshpoints):
        mesh_node=probe_start+j*probe_stride

        spike_array=np.asarray(track_probes[mesh_node][0].data[:][0])
        track_nz=np.nonzero(track_probes[mesh_node][0].data[:][0])
        logging.info('mesh_node {}: {}'.format(j, track_nz))
        logging.info('mesh_node {} buffer: {}'.format(j, np.nonzero(track_probes[mesh_node+2][0].data[:][0]) ) )
        
        bins=track_nz[0]
        bin_start=0
        num_bins=np.size(track_nz)
        track_j=track_probes[mesh_node][0].data[:][1]
        for i in range(0, num_bins):
            bin_end=bins[i]
            count[j,i]=np.sum(track_j[bin_start:bin_end])
            bin_start=bin_end+1

    k=0
    logging.info('super_1: {}'.format( np.nonzero(track_probes[0][0].data[:])) )
    logging.info('super_2: {}'.format( np.nonzero(track_probes[0][1].data[:])) )
    for j in range(num_meshpoints*5+probe_start, len(track_probes)-2):
        logging.info('super_local {}_1: {}'.format(k, np.nonzero(track_probes[j][0].data[:][0]) ) )
        logging.info('super_local {}_2: {}'.format(k, np.nonzero(track_probes[j][0].data[:][1]) ) )
        k=k+1
        k1=j
    
    np.set_printoptions(threshold=sys.maxsize)
    logging.info('super_local_U2: {}'.format( track_probes[k1+1][0].data[:][1][25000:30000]) ) 
    logging.info('super_local_U3: {}'.format( track_probes[k1+2][0].data[:][1][25000:30000]) )
    return count, num_bins

def process_outputs_split(track_probes, probe_start=0, probe_stride=1, num_meshpoints = 41):
    #####
    # Script to convert probe tracks into a density over time for each tracked mesh point
    # 
    # This applies to Track = 6, where supervisors and mesh nodes are separated
    #
    # Probes are typically ordered as follows
    #
    # [0][0-1]: Supervisor probes, Supervisor FindZero probes
    # [1][0-...] Mesh point '0' - Mesh point 'i'.  Only counter neuron spike probes
    # 
    # Probe start is set in case a specific set of probes is needed to be monitored
    # Probe stride here in case a specific set of probes is needed to be monitored
    
    
    # Determine the end times of each model update step
    #     This is because model updates are non-deterministic length in simulation timesteps

    super_probes = track_probes[0]
    mesh_probes = track_probes[1]
                
    bins=np.nonzero(super_probes[0][1].data[:][0])
    num_bins=np.size(bins)
    count=np.zeros((num_meshpoints, num_bins+1))
    if(num_meshpoints>0):
        for j in range(0, num_meshpoints):
            mesh_node=probe_start+j*probe_stride
            spike_array=np.asarray(mesh_probes[mesh_node][1].data[:][0])
            track_nz=np.nonzero(mesh_probes[mesh_node][1].data[:][0])
            bins=track_nz[0]
            bin_start=0
            num_bins=np.size(track_nz)
            track_j=mesh_probes[mesh_node][1].data[:][1]
            for i in range(0, num_bins):
                bin_end=bins[i]
                count[j,i]=np.sum(track_j[bin_start:bin_end])
                bin_start=bin_end+1
    return count, num_bins
