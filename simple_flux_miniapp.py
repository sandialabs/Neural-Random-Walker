# Copyright 2023 National Technology & Engineering Solutions of Sandia, LLC (NTESS). 
# Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains 
# certain rights in this software.
# Apache License Version 2.0, January 2004 http://www.apache.org/licenses/
from __future__ import absolute_import
import os
os.environ['SLURM'] = "1"
import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import sys
import argparse
import logging
from datetime import datetime
import time
import utils.logging_utils as utils
import signal
import random

def signal_handler(signum, frame):
    rw_logger.info('Signal handler called with signal {}'.format(signum))
    os.killpg(os.getpid(),signum)
    sys.exit(0)

signal.signal(signal.SIGTERM, signal_handler)
signal.signal(signal.SIGINT, signal_handler)

rw_logger = logging.getLogger("NRW")

def run_fugu(matrix_filename = None,
            neuraltime = None,
            initial_walkers = None,
            meshSize = None,
            runtime_args = {}):
    from densityrw.rw_miniapp_fugu import run_miniapp
    fugu_backend = runtime_args['fugu_backend']
    use_sinks = runtime_args['use_sinks']

    # In this section, we need to call a function to run the simulation on
    # FUGU.  We need to run the simulation for each starting location in
    # the meshpoints array.
    # First, we get the spike train results of running FUGU. This output
    # is a pandas dataframe. To get these results, we tell FUGU that
    # we have a transition matrix named 'matrix_filename', we tell FUGU
    # how many neural time steps to use, 'neuraltime'. We also give it a 
    # tuple in the form of 'initial_walkers'.  This takes the form [i,M] 
    # telling it to start M walkers on node i. Next we specify whether to
    # 'use_sinks'.  If we use sinks - True - then we keep track of walkers
    # in the absorbing node (ie, the node that leaves the domain).  
    # Otherwise - False - we do not track walkers that enter the absorbing 
    # node. Finally, we indicate what fugu_backend we want to use.
    results = run_miniapp(matrix_filename, neuraltime, initial_walkers, use_sinks, fugu_backend)
    # The output is a pandas datafile.  It has three columns.  One of
    # these columns is called 'time.'  This column is an integer corresponding
    # to the neural time step.  A model time step occurs whenever the
    # difference between successive timesteps is greater than 1.  We add
    # a column to the datafile corresponding to the model time step.
    # First, we grab the array of neural time steps.
    time_array = np.array(results['time'])
    # Next, we get an array that is the difference of the successive terms.
    diff = time_array[1:]-time_array[0:-1]
    # Now we set the timestep numbers by creating an array with one more location
    # than the difference array (one location is lost by the subtraction), ...
    timestep = np.zeros((len(diff)+1,))
    # ... then setting the first time step that was lost to zero ...
    timestep[0]=0
    # ... and setting the rest of the time steps appropriately.
    timestep[1:]=np.cumsum(diff>1)
    # Now, we add the time step information to the results pandas datafile,
    # as integers.
    results['timestep']=timestep
    results['timestep']=results['timestep'].astype('int')
    # The last value in the timestep array is the number of model time
    # steps simulated.  We save this number.
    NT = int(timestep[-1])

    # Now that we have added an additional column to the datafile detailing
    # the time step of the simulation, we now change the 'node' column of the
    # datafile so that it just gives the integer node number, and not a string
    # tuple.
    results['node']=results['node'].map(lambda x: eval(x)[0])

    # With the prep work done, we can create a temporary array to detail
    # the number of walkers on any given node at a particular time step.
    tempdata = np.zeros((meshSize,NT))
    # We now fill this array with a loop. We iterate through the unique nodes
    # hit (we don't want to do the extra work on nodes that never are hit),
    # and then iterate through the time steps.  We then find the size of the
    # portion of the data file that has an identical node and time step value.
    # This is the number of times a particular node was occupied on a given
    # time step.
    for node in range(meshSize): #results['node'].unique():   ## Note results contains the outside region
        for tstep in range(NT): #results['timestep'].unique():
            tempdata[node,tstep] = len(results[np.logical_and(results['node']==node,results['timestep']==tstep)])

    return tempdata

def run_loihi(matrix_filename = None,
            neuraltime = None,
            initial_walkers = None,
            meshSize = None,
            runtime_args = {}):
    from loihirw.rw_model import rad_test
    from loihirw.rw_analysis import process_outputs_split
    assert len(initial_walkers) == 1
    if 'core_density' in runtime_args:
        core_density = runtime_args['core_density']
    else:
        core_density = 1.1
    if 'input_times' in runtime_args:
        input_times = runtime_args['input_times']
    else:
        input_times = [1]
    if 'track' in runtime_args:
        track = runtime_args['track']
    else:
        track = 6

    size = (meshSize, 1)  #In this miniapp we are always 1D
    i = initial_walkers[0][0]
    num_walkers = initial_walkers[0][1]
    rad_test_results = rad_test(matrix_filename,
                                [i],
                                neuraltime,
                                num_walkers=num_walkers,
                                core_density=core_density,
                                verbose = 0,
                                input_times = input_times,
                                track=track,
                                size = size)
    tprobe1, eprobe1, track_probes, core_neuroncount, max_coreid = rad_test_results

    # The way this will be configured is as follows:
    #   track_probes[0] -- supervisor nodes
    #   track_probes[1] -- mesh nodes
    
    # within each of these, we will have the following
    #   track_probes[xx][group ID][0]=title
    #   track_probes[xx][group ID][1]=probe data
    #       where groupID will be the supervisor group OR the node ID
    # 
    # within each of these you can access the data by the following
    #   track_probes[xx][yy][1].data[:][neuron ID]
    #       This is the 1xNUMTIME stream of SPIKES (or other signal that probe measures)
    
    count, num_bins = process_outputs_split(track_probes, 0, 1, meshSize)

    return count

def run_spinnaker(matrix_filename = None,
            neuraltime = None,
            initial_walkers = None,
            meshSize = None,
            runtime_args = {}):
    assert len(initial_walkers) == 1
    import spynnaker8 as pynn
    from spinnakerrw import randomwalker as rw
    if 'rand_seed' in runtime_args:
        rand_seed = runtime_args['rand_seed']
    else:
        rand_seed = 1
    i = initial_walkers[0][0]
    M = initial_walkers[0][1]
    generators, supervisors = rw.setup_network(pynn,
                                               num_time = neuraltime,
                                               num_walkers = M,
                                               starting_node = i,
                                               matrix_filename = matrix_filename,
                                               seed = rand_seed)
    output_bins, ct_bins = rw.analyze_rw(generators, supervisors, meshSize+1, neuraltime)
    pynn.end()
    return output_bins

def run_pc(matrix_filename = None,
           neuraltime = None,
           initial_walkers = None,
           meshSize = None,
           runtime_args = {}):
    myQ = np.genfromtxt(matrix_filename,delimiter=',')
    # This is meant to test a python implementation. If this is the mode that
    # gets run, we will assume neural time is the model time step.
    tempdata = np.zeros((2,neuraltime))
    # We set the initial condition
    place = initial_walkers[0][0]
    amount = initial_walkers[0][1]
    tempdata[place,0] = amount

    # Now we simulate
    for i in range(1,neuraltime):
        for j in range(0,int(tempdata[0,i-1])):
            testrand = np.random.uniform()
            if testrand<=myQ[0,0]:
                tempdata[0,i] = tempdata[0,i]+1
            elif testrand<=(myQ[0,0]+myQ[0,1]):
                tempdata[1,i] = tempdata[1,i]+1
        for j in range(0,int(tempdata[1,i-1])):
            testrand = np.random.uniform()
            if testrand<=myQ[1,0]: # Note the change in Q matrix line.
                tempdata[0,i] = tempdata[0,i]+1
            elif testrand<=(myQ[1,0]+myQ[1,1]):
                tempdata[1,i] = tempdata[1,i]+1
    
    return tempdata

run_functions = {'fugu':run_fugu,'loihi':run_loihi,'spinnaker':run_spinnaker,'pc':run_pc}

def run(args):
    rmode = args['run_mode']
    M = args['num_walks']
    dt = args['timestep']
    initp = args['init_pos']
    initn = args['init_neg']
    SigS = args['SigS']
    SigA = args['SigA']
    max_time = args['max_time']
    neuraltime = args['neural_timesteps']
    matrix_filename = args['matrix_filename']
    fugu_backend = args['fugu_backend']
    results_filename = args['results_filename']
    plotr = args['plot_results']
    use_sinks = args['use_sinks']
    rand_seed = args['rand_seed']

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

    # Before any simulation can occur, we must create a transition matrix
    # for the problem.  To do so, we perform checks on the input parameters.
    # This ensures that no more than a single event can occur per time step.

    pone = (SigS+SigA)*dt*(np.exp(-(SigA+SigS)*dt))
    pzero = np.exp(-(SigA+SigS)*dt)
    if (1-pone-pzero)>0.01:
        # We choose to not allow bad parameter choices leading to bad
        # expectations on the simulation.  However, this check can be
        # removed if a simulation with 'bad' parameters is desired.
        rw_logger.error("The probability of more than one scattering event occurring in a single time step is greater than 0.01.")
        rw_logger.error("Alter your choice of SigA, SigS, or dt so that 1-(SigA+SigS)*dt*exp(-(SigA+SigS)*dt)-exp(-(SigA+SigS)*dt)<=0.01.")
        raise ValueError("The probability of more than one scattering event occurring in a single time step is too large.")

    # If our parameters pass the test, we can move on with making the
    # transition matrix. To do so, we determine a few probabilities.
    pscatter = pone*(SigS/(SigA+SigS))
    pabsorb = pone*(SigA/(SigA+SigS))

    # We now initialize our transition matrix. This simple problem has three
    # possible states: positive direction, negative direction, and absorbed.
    # So we count two real states and then add an extra for the absorption.
    meshSize = 2
    myQ = np.zeros((meshSize+1,meshSize+1))

    # We now start to fill out the matrix. Since there are only three states,
    # we write them all explicitly.
    
    # Once a scattering event occurs, we assume that the new direction is 
    # chosen uniformly. That is, no matter the previous direction, once a
    # scattering event happens, the probability of getting the positive or
    # negative direction is 1/2.
    
    # A particle transitions back to itself if (1) a scattering event happens
    # and the new direction chosen is the same as the old; or (2) no event
    # no event happens.
    myQ[0,0] = pscatter*0.5 + 1-pone
    # A particle transitions to the other state only if a scattering event
    # happens and the other direction is chosen.
    myQ[0,1] = pscatter*0.5
    # A particle can only be absorbed if an absorption event happens
    myQ[0,2] = pabsorb
    
    # We make similar states for the next one.
    myQ[1,0] = pscatter*0.5
    myQ[1,1] = pscatter*0.5 + 1-pone
    myQ[1,2] = pabsorb
    
    # The only posibility for an absorbed particle is to stay absorbed.
    myQ[2,2] = 1

    # The matrix we built should have rows that sum to 1, but due to
    # machine rounding errors, it might not. Therefore, we do a quick
    # normalization of the rows.
    myQ = myQ/myQ.sum(axis=1)[:,None]
    
    # Now that we've done our check on the row sums, we secretly want the 
    # absorption probability to be zero so that we discard particles that
    # end up in this state.
    myQ[2,2]=0

    
    # Now, we save this matrix as $matrix_filename. This is given to the
    # next programs.
    np.savetxt(matrix_filename, myQ, delimiter=",")
    rw_logger.info("Transition matrix construction complete")
    rw_logger.info("    matrix saved to {}".format(matrix_filename))

    # This is a time based solution. To prevent dynamic arrays, we will 
    # preallocate a bit of space for the solution.
    if rmode=='pc':
        maxT=min(neuraltime,int(max_time/dt))+2
    else:
        maxT = int(max_time/dt)+2
    rwSol = np.zeros((2,maxT))
    # We now need to condition on the rmode to determine which simulation to
    # run.

    run_function = run_functions[rmode]

    rw_logger.info("Beginning simulation")
    rw_logger.info("    using {} on {}".format(rmode,datetime.now()))
    for i in range(0,meshSize):
        t0 = time.time()
        tempdata = run_function(matrix_filename = matrix_filename,
                                neuraltime = neuraltime,
                                initial_walkers = [[i,M]],
                                meshSize = meshSize,
                                runtime_args = {'fugu_backend': fugu_backend,
                                                'use_sinks': use_sinks,
                                                'rand_seed': rand_seed,
                                                } )
        rw_logger.info('    mesh point: {}'.format(i))
        rw_logger.debug('        elapsed time: {}'.format(time.time() - t0))
        # Now that the counts are complete, we can calculate the solution for this
        # node. 
        # FWe need to iterate through completed timesteps. We throw out the
        # last two timesteps since these might not complete.
        maxT = min((tempdata.shape[1]-2),maxT)
        for j in range(0,maxT):
            # We calculate the solution value.
            rwSol[i,j]=(initp*tempdata[0,j]+initn*tempdata[1,j])/M
        
        # This calculates the entire solution at all timepoints for node i.

    rw_logger.info("Simulation using {} completed on {}".format(rmode, datetime.now()))

    # Now that the solution has been calculated, we can plot it, if so desired.
    if plotr:
        import matplotlib.pyplot as plt
        # We will plot against the true solution. To do so, we calculate the
        # real solution.
        trueSol = np.zeros((2,maxT))
        tarray = np.zeros(maxT)
        for i in range(0,maxT):
            tarray[i] = i*dt
            trueSol[0,i] = (initp/2)*(np.exp(-SigA*i*dt)+np.exp(-(SigA+SigS)*i*dt)) + (initn/2)*(np.exp(-SigA*i*dt)-np.exp(-(SigA+SigS)*i*dt))
            trueSol[1,i] = (initp/2)*(np.exp(-SigA*i*dt)-np.exp(-(SigA+SigS)*i*dt)) + (initn/2)*(np.exp(-SigA*i*dt)+np.exp(-(SigA+SigS)*i*dt))
        rw_logger.info("Plotting solution")
        f1 = plt.figure(1)
        plt.plot(tarray,trueSol[0,:],color='k',label='Analytic',linestyle='dashed')
        plt.plot(tarray,trueSol[1,:],color='k',linestyle='dashed')
        plt.plot(tarray,rwSol[0,0:maxT],color='r',label='Positive')
        plt.plot(tarray,rwSol[1,0:maxT],color='b',label='Negative')
        plt.xlabel("Time")
        plt.ylabel("Solution")
        plt.legend()
        plt.close(f1)
        figname = results_filename.split(sep='.')[0]+'_plot.png'
        f1.savefig(figname)
        rw_logger.info("Plot saved to {}".format(figname))
    # We also save the data.
    np.savetxt(results_filename,rwSol,delimiter=",")
    rw_logger.info("Solution saved to {}".format(results_filename))

if __name__ == "__main__":
    # This script will create a transition matrix for a simple two state fluence 
    # problem with the following described variables.  A transition matrix will be
    # calculated based on these variables and saved as $matrix_filename. Once the 
    # matrix is created, random walkers will be simulated starting on every position
    # of the mesh. These will be simulated according to user specification.

    # There are several options, described below.  Some variable are required for the
    # simulation to run.  These are M, dt, ip, in, SigS, and SigA.  M is the number
    # of walkers to start on each position in the mesh.  dt is the time discretization
    # size.  ip and in are the initial condition for the positive and negative solutions
    # respectively. SigS is the scattering cross-section (or rate) and SigA is the 
    # absorption cross-section (or rate).

    # The problem approximates the fluence for two states in a two state problem over
    # time.  It can be though of as providing a function F(t,1) and F(t,-1) where
    # 1 is the positive direction state and -1 is the negative direction state. It is
    # based off of SN3.8 from the Nature Electronics paper located here:
    # https://www.nature.com/articles/s41928-021-00705-7
    
    # The default values in this simulation are different from the referenced
    # paper. They have been selected to best approximate the solution with the
    # Loihi hardware.

    # Variables are taken as options from the command line and are parsed using argparse.
    # The details and defaults of these variables are described below in the Usage section.
    # Please note that balancing and optimizng the parameters is a challenging process.
    # They must be selected so that the discrete time Markov chain well approximates the
    # underling stochastic process and selected so that the lower precision probability
    # on Loihi introduces negligible error.
    parser = argparse.ArgumentParser()
    parser.add_argument('-r','--run_mode', default='fugu', choices=['fugu','loihi','spinnaker','pc'], help='The mode specifies how the random walks are ran. Must be one of fugu, loihi, spinnaker, or pc. The default is fugu. Specifying pc gives a python implementation.')
    parser.add_argument('-M','--num_walks',type=int,default=500,help='The number of walkers to start on each location in the mesh.  The default is 500.')
    parser.add_argument('-dt','--timestep',type=float,default=0.005,help='The time step of the simulation. The default is 0.005.')
    parser.add_argument('-ip','--init_pos',type=float,default=5.0,help='The initial value for the positive direction solution. The default is 5.')
    parser.add_argument('-in','--init_neg',type=float,default=3.0,help='The initial value for the negative direction solution. The default is 3.')
    parser.add_argument('-ss','--SigS',type=float,default=8.0,help='The scattering cross-section of the particle. This is the rate at which scattering events occur. The default is 8.0.')
    parser.add_argument('-sa','--SigA',type=float,default=2.0,help='The absorption cross-section.  This is the rate at which particles are absorbed. The default is 2.0.')
    parser.add_argument('-mt','--max_time',type=float,default=5.0,help='The absolute max model time of simulation. This is the largest value of time that could be displayed in the solution. The actual largest model time of simulation may be smaller than this value if the neural timestep is too small for Loihi or Fugu->Loihi simulations. For conventional (pc) simulations, the actual time may be smaller if dt*nt<mt.')
    parser.add_argument('-nt','--neural_timesteps',type=int,default=500000,help='The number of neural timesteps the simulation is run for. This should be MUCH greater than the number of model time steps desired.  The default is 500000. For pc, this is the number of model timesteps.')
    parser.add_argument('--debug',default=logging.WARNING,action="store_const",dest="loglevel",const=logging.DEBUG,help="Print logging DEBUG statements.",)
    parser.add_argument('--verbose',action="store_const",dest="loglevel",const=logging.INFO,help="Print logging INFO statements.")
    parser.add_argument('--matrix_filename',default='curmat.csv',help='Filename for temporary storage of transition matrix.')
    parser.add_argument('--results_filename',default='rwSol.csv',help='Filename to record the results.')
    parser.add_argument('--fugu_backend',default='snn',choices=['snn'],help='Runs a spiking simulator using Fugu.')
    parser.add_argument('--plot_results',default=False,action='store_true',help='Plots the results after finishing.')
    parser.add_argument('--log_file',default='~/.miniapps/simple_flux_miniapp.log',help='Location of the log file.')
    parser.add_argument('--rand_seed', type=int, default=-1, help="Sets the seed for the random number generator. Otherwise uses a time-based 'random' RNG seed.")
    parser.add_argument('--use_sinks',default=False,action='store_true',help='Include sinks in a fugu simulation; default is False.')


    args = parser.parse_args()
    args = vars(args)

    if args['rand_seed'] < 0:
        # We will use a time-based "random" RNG seed
        seed = datetime.now().microsecond+datetime.now().second+datetime.now().minute*60+datetime.now().hour*24
    else:
        seed = args['rand_seed']

    np.random.seed(seed)
    random.seed(seed)

    # Check if nxsdk is available for running on loihi
    try:
        import nxsdk
    except ImportError as e:
        if args['run_mode'] == 'loihi' or args['fugu_backend'] == 'loihi' :
            raise SystemExit('\n *** NxSDK package is not installed. Running RandomWalk on Loihi is unavailable. *** \n')
        else:
            pass

    # Set up logging.
    log_file = os.path.expanduser(args['log_file'])
    # Create .miniapps if it doesn't exist
    if not os.path.exists(os.path.dirname(log_file)):
        os.makedirs(os.path.dirname(log_file))

    utils.setup_logging(log_file=log_file,useLogFile=True,log_level=args['loglevel'])

    rw_logger.info('Using config: {}'.format(args))
    rw_logger.info('Starting run on {}'.format(datetime.now())  )

    run(args)
    rw_logger.info('Completed run on {}'.format(datetime.now()) )
