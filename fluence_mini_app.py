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

def sourceTerm(x,y):
    # This function is the source term for the problem.  We could make this
    # variable based on inputs from the main script, but instead we will
    # make this a static function.  It takes as input a position, angle pair
    # given by (x,y).  It returns the value of the source function at that
    # point.
    val = (x**2) + ((y+(7.0/16.0)**2))
    if (val <= 0.25):
        aaa = 3.0
    else:
        aaa = 0.0
    return aaa

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
    if 'runclean' in runtime_args:
        runclean = runtime_args['runclean']
    else:
        runclean = 1

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
    
    # This will produce a [meshSize x numbins] sized matrix of the walker counts at each mesh point for that activation
    # 
    #

    if(runclean == 1):
        nz_checka=np.nonzero(track_probes[0][0][1].data[:][0])
        nz_checkb=np.nonzero(track_probes[0][0][1].data[:][1])
        nz_check=np.nonzero(track_probes[0][1][1].data[:][2])
        rw_logger.info("First empty number matrix: {}".format(nz_check))
        rw_logger.info("First empty number: {}".format(nz_check[0][0]))
        superdiff=[]
        for j in range(0, len(nz_checkb[0])):
             superdiff.append(nz_checkb[0][j]-nz_checka[0][j])
        rw_logger.info("Supervisor difference: {}".format(superdiff))
        rw_logger.info("Count: {}".format(count))
        rw_logger.info("Number of bins: {}".format(num_bins))
        # We will now re-run the simulation without tracking for the length identified above
        rad_test_results = rad_test([i],
                                nz_check[0][0],
                                num_walkers=num_walkers,
                                core_density=core_density,
                                verbose = 0,
                                input_times = input_times,
                                track=0,
                                custom_network = matrix_filename)
        tprobe1_7, eprobe1_7, track_probes_7, core_neuroncount_7, max_coreid = rad_test_results

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
    probvecs = np.cumsum(myQ,axis=1)
    # This is meant to test a python implementation. If this is the mode that
    # gets run, we will assume neural time is the model time step.
    tempdata = np.zeros((meshSize+1,neuraltime))
    # We set the initial condition
    place = initial_walkers[0][0]
    amount = initial_walkers[0][1]
    tempdata[place,0] = amount

    # Now we simulate.  The initial condition is set, so we iterate through
    # all forward time assuming the neural timestep is one standard model
    # timestep
    for i in range(1,neuraltime):
        # For the current time step, we iterate through all locations in the
        # mesh.  For each location, we simulate all walkers that existed on the
        # location in the previous timestep forward one timestep. Note, the
        # absorbing node is not simulated.
        for j in range(0,meshSize):
            for k in range(0,int(tempdata[j,i-1])):
                testrand = np.random.uniform()
                tnsp = np.searchsorted(probvecs[j,:],testrand)
                tempdata[tnsp,i] += 1
    
    return tempdata

run_functions = {'fugu':run_fugu,'loihi':run_loihi,'spinnaker':run_spinnaker,'pc':run_pc}

def run(args):
    rmode = args['run_mode']
    M = args['num_walks']
    dt = args['timestep']
    da = args['anglestep']
    v = args['velocity']
    SigS = args['SigS']
    SigA = args['SigA']
    L = args['interval_length']
    neuraltime = args['neural_timesteps']
    matrix_filename = args['matrix_filename']
    fugu_backend = args['fugu_backend']
    results_filename = args['results_filename']
    plotr = args['plot_results']
    use_sinks = args['use_sinks']
    rand_seed = args['rand_seed']
    runclean = args['runclean']

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

    # Before any simulation can occur, we must create a transition matrix
    # for the problem.  We start by determining the spatial discretization
    # size, a dependent choice based on previous variables.  This parameter
    # is chosen so that no "rounding" errors will occur when particles are
    # traveling.  That is, it is chosen so that if a particle starts at the
    # midpoint of a spatial discretization bin with an angle from the allowed
    # angle choices, that it will always end exactly at the midpoint of another
    # spatial discretization bin.
    ds = da*v*dt/2

    # Next, we calculate the probability that with the given parameters that
    # more than a single scattering event could occur in a single time step.
    pone = v*SigS*dt*(np.exp(-v*SigS*dt))
    pzero = np.exp(-v*SigS*dt)
    if (1-pone-pzero)>0.05:
        # We choose to not allow bad parameter choices leading to bad
        # expectations on the simulation.  However, this check can be
        # removed if a simulation with 'bad' parameters is desired.
        rw_logger.error("The probability of more than one scattering event occurring in a single time step is greater than 0.05.")
        rw_logger.error("Alter your choice of v, SigS, or dt so that 1-v*SigS*dt*exp(-v*SigS*dt)-exp(-v*SigS*dt)<=0.05.")
        raise ValueError("The probability of more than one scattering event occurring in a single time step is too large.")

    # If our parameters pass the test, we can move on with making the
    # transition matrix.  To do so, we construct some arrays.  The first
    # is the spatial array.  These define the end points of the bins,
    # made by the divisions due to the discretization size ds.
    NS = int((2*L)/ds)+1 # The number of divisions.
    space = np.linspace(-L,L,NS) # The spatial array.
    # Next, we will find the centerpoints of these divisions. To do so,
    # we increment the spatial array by ds/2 ...
    cps = (space+(ds/2))
    # ... and then we get rid of the final position since it isn't the
    # mid point of one of our bins.
    cps = cps[0:(NS-1)]
    # Next, we must determine the angular direction array. This is similar
    # to the spatial array construction.
    ND = int(2/da)+1
    dirs = np.linspace(-1,1,ND)
    # We similarly find the midpoints, and use these as our allowed directions.
    directions = (dirs+(da/2))
    directions = directions[0:(ND-1)]


    # Now, we create the array of (cps,directions) tuples to random walk over.
    # This array is critical not only for the construction of the transition
    # matrix, but also for interpreting the returned data into a solution.
    meshSize = (NS-1)*(ND-1)
    rw_logger.info('Mesh Size: {}'.format(meshSize))
    meshpoints = np.zeros((meshSize,2))
    for j in range(0,NS-1):
        for i in range(0,ND-1):
            row = i + j*(ND-1)
            meshpoints[row,0]=cps[j]
            meshpoints[row,1]=directions[i]

    # We now determine a few probabilities based on our earlier calculation
    # of a single event.  Remember, after a scattering event, the particle
    # will choose a new direction UNIFORMLY from the list of allowed directions.
    # Therefore, the probability of any taking on a *NEW* direction in any single
    # timestep is:
    freshprob = pone/(ND-1)
    # The probability of keeping the same direction after any given timestep
    # is different. You can keep your direction by either having a scattering event
    # and uniformly choosing your same direction from the list of allowable
    # directions, or you can not have a scattering event. We have assumed that the
    # probability of having more than a single scattering event in a time interval is
    # small enough, and we performed a check on this. Therefore, we will assume that
    # (1-pone) is the probability of NO scattering events. Hence, the probability of
    # keeping your same direction in a time step is:
    returnprob = freshprob + (1-pone)

    # We now initialize our transition matrix.  It will have an extra
    # position for escape from the domain [-L,L].
    myQ = np.zeros((meshSize+1,meshSize+1))

    # We now start to fill out the matrix. We have a special case:
    # the escape state is absorbing.
    myQ[meshSize,meshSize]=1

    # Now we handle the other states in a loop
    for i in range(0,meshSize):
        # First determine the new position of the particle
        newpos = meshpoints[i,0]-v*meshpoints[i,1]*dt
        # Now we check to see if we left the wire
        if abs(newpos)>L :
            # This mesh point will always leave the wire
            myQ[i,meshSize]=1
        else:
            # Otherwise, the particle remains in the wire.  Determine
            # the location of the new position
            newindex = np.where((np.round(cps-newpos,8))==0)
            newindex = newindex[0][0]
            for j in range(0,ND-1):
                # Assign the probability to all transitions to the new
                # position with all directions.
                transition = j + (newindex*(ND-1))
                myQ[i,transition] = freshprob
            # Add in the probability of no scattering events to just a
            # transition in space and no change in direction
            samedir = np.where((np.round(directions-meshpoints[i,1],8))==0)
            samedir = samedir[0][0]
            sameplace = samedir+(newindex*(ND-1))
            myQ[i,sameplace] = returnprob

    # The matrix we built should have rows that sum to 1, but due to
    # machine rounding errors, it might not. Therefore, we do a quick
    # normalization of the rows.
    myQ = myQ/myQ.sum(axis=1)[:,None]

    # We want to set the escape state as eliminating the walkers
    myQ[meshSize,meshSize]=0
    # Now, we save this matrix as $matrix_filename. This is given to the
    # next programs.
    np.savetxt(matrix_filename, myQ, delimiter=",")
    rw_logger.info("Transition matrix construction complete")
    rw_logger.info("    matrix saved to {}".format(matrix_filename))

    # To prep for the solution, we will create an array to stor values.
    rwSol = np.zeros((NS-1,ND-1))
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
                                                'runclean': runclean,
                                                'rand_seed': rand_seed,
                                                } )

        rw_logger.info('    mesh point: {}'.format(i))
        rw_logger.debug('        elapsed time: {}'.format(time.time() - t0))
        # Now that the counts are complete, we can calculate the solution for this
        # node. First we determine what (position,angle) pair is given by the
        # current node i. This is just a matter of undoing the embedding into the
        # meshpoints order.
        aind = i%(ND-1)
        if (aind==0):
            aind = ND-1
        aind = aind - 1
        posind = int((i-aind)/(ND-1))
        # Next, we start an accumulator variable and iterate through the counts
        # we just made in order to save the solution.
        tempval = 0
        # First we iterate through the time steps, skipping the initial timestep
        # and skipping the final time step. The final time step might not finish,
        # and the initial condition, or first time step, isn't used in the
        # solution approximation.

        for j in range(1,tempdata.shape[1]-1):
            # Then we iterate through the nodes.
            for k in range(0,meshSize):
                # We only want to do our calculation if walkers exist on the
                # given node at the given time.
                if (tempdata[k,j]!=0):
                    tempval = tempval + tempdata[k,j]*np.exp(-SigA*dt*j)*v*sourceTerm(meshpoints[k,0],meshpoints[k,1])*dt

        # After calculating our solution value, we normalize it by M and assign
        # it to the solution array.
        rwSol[posind,aind] = tempval/M

    rw_logger.info("Simulation using {} completed on {}".format(rmode, datetime.now()))

    # Now that the solution has been calculated, we can plot it!
    if plotr:
        plot_results(rwSol,dirs,space,results_filename)
    # We also save the data.
    np.savetxt(results_filename,rwSol,delimiter=",")
    rw_logger.info("Solution saved to {}".format(results_filename))

def plot_results(data,angles,spaces,rfilename):
    import matplotlib.pyplot as plt
    rw_logger.info("Plotting solution")    
    fig, ax = plt.subplots()
    im = ax.pcolormesh(angles,spaces,data)
    fig.colorbar(im)
    ax.set_title('Angular Fluence')
    ax.set_ylabel('Position')
    ax.set_xlabel('Angle')
    plt.close()
    figname = rfilename.split(sep=".")[0]+"_plot.png"
    plt.savefig(figname)
    rw_logger.info("Plot saved to {}".format(figname))

if __name__ == "__main__":
    # This script will create a transition matrix for an angular fluence problem
    # with the following variables.  The matrix will be saved as $matrix_filename.
    # Once the matrix is created, FUGU will run walkers starting on every position
    # in the mesh.

    # There are several options, described below.  Some variables are required for the
    # simulation to run.  These are M, dt, da, v, SigS, SigA, and L.  M is the number
    # of walkers to start on each position in the mesh.  dt is the time discretization
    # size.  da is the angular discritization size - this corresponds to the different
    # 'direction' values the random walkers can assume.  v is the maximum velocity of
    # the particle, achieved only when the particle has an angle of 1 (or -v when the
    # particle has an angle of -1). SigS is the scattering cross-section (or rate) and
    # SigA is the absorption cross-section (or rate).  L specifies the interval [-L,L]
    # that the particle is allowed to exist in.

    # The 'original' problem from the Nature Electronics paper used the following
    # values:
    # M = 1000
    # dt = 0.01
    # da = 0.0666667 (=1/15)
    # v = 200
    # SigS = 0.15
    # SigA = 0
    # L = 1
    # In addition, the 'original' Nature Electronics paper used a different source term.
    # Specifically, S(x,w)=0.015 whenever |x|<0.5.  Here, x is position and w is angle.


    # Variables are taken as options from the command line and are parsed using argparse.
    parser = argparse.ArgumentParser()
    parser.add_argument('-r','--run_mode', default='fugu', choices=['fugu','loihi','spinnaker','pc'], help='The mode specifies how the random walks are ran. Must be one of fugu, loihi, spinnaker, or pc. The default is fugu. Specifying pc gives a python implementation.')
    parser.add_argument('-M','--num_walks',type=int,default=100,help='The number of walkers to start on each location in the mesh.  The default is 100.')
    parser.add_argument('-dt','--timestep',type=float,default=0.01,help='The time step of the simulation. The default is 0.01.')
    parser.add_argument('-da','--anglestep',type=float,default=1.0/15.0,help='The angular discretization step size. The default is 1/15.')
    parser.add_argument('-v','--velocity',type=float,default=200,help='The maximum velocity of the particle. The default is 200.')
    parser.add_argument('-ss','--SigS',type=float,default=0.15,help='The scattering cross-section of the particle. This is the rate at which scattering events occur. The default is 0.15.')
    parser.add_argument('-sa','--SigA',type=float,default=0,help='The absorption cross-section.  This is the rate at which particles are absorbed. The default is zero.')
    parser.add_argument('-L','--interval_length',type=float,default=1,help='The length of the spatial interval [-L,L] the particle is allowed to travel through.  The default is 1.')
    parser.add_argument('-nt','--neural_timesteps',type=int,default=30000,help='The number of neural timesteps the simulation is run for. This should be MUCH greater than the number of model time steps desired.  The default is 30000. For pc run mode, this is the number of model timesteps.')
    parser.add_argument('-d','--debug',default=logging.WARNING,action="store_const",dest="loglevel",const=logging.DEBUG,help="Print logging DEBUG statements.",)
    parser.add_argument('--verbose',action="store_const",dest="loglevel",const=logging.INFO,help="Print logging INFO statements.")
    parser.add_argument('--matrix_filename',default='curmat.csv',help='Filename for temporary storage of transition matrix.')
    parser.add_argument('--results_filename',default='rwSol.csv',help='Filename to record the results.')
    parser.add_argument('--fugu_backend',default='snn',choices=['snn'],help='Runs a spiking simulator using Fugu.')
    parser.add_argument('--plot_results',default=False,action='store_true',help='Plots the results after finishing.')
    parser.add_argument('--log_file',default='~/.miniapps/fluence_miniapp.log',help='Location of the log file.')
    parser.add_argument('--runclean', default=0,help='Run probe-free replicate in a Loihi run.')
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
