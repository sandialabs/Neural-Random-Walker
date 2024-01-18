# Installation
Here are instructions to setup a python environment to run the Random Walk program using Anaconda. The instructions below will create a stand alone environment capable of running the random walk code. To install in an exhisting environment, manually install the packages found inside `rw_requirements.yml`.

## Part 1 Pull Fugu Repository
This repository comes with the Fugu repository. Before we can setup the python environment, we must pull from the fugu repository. This task is accomplish by running the below commands from a terminal.
```
git submodule init
git submodule update
```

## Part 2 Create Conda Environment
We assume that you have anaconda distribution already installed on your machine. Again from a terminal run these commands from inside the random walk repository root directory.
```
conda env create --file rw_requirements.yml
conda activate rwalk-fugu
conda develop $PWD/Fugu
```
This part creates a `rwalk-fugu` conda environment and installs the necessary packages for the Fugu submodule. Additionally, it adds the Fugu submodule to the `rwalk-fugu` conda environment path.

Note: The `rw_requirments.yml` creates an environment called `rwalk-fugu` by default. To change this behavior modify the `name` property to the desired environment name.

## Part 3 NxSDK Install (Optional)
This part is required if you plan to run the program on Loihi hardware. It requires the NxSDK python package. From a terminal, run these commands
```
tar -xvf nxsdk-<version>.tar.gz
nxsdk-apps-<version>.tar.gz
```
Update `/PATH/TO/nxsdk-<version>/requirements.txt` file to:
```
attrdict>=2.0.1
numpy==1.15.*
pandas==1.0.*
matplotlib>=2.2.2
imageio>=2.6.1
scikit-image>=0.14.2
scipy==1.5.*
scikit-learn>=0.19.2
jinja2>=2.10
coloredlogs>=10.0
grpcio>=1.19.0
protobuf==3.19.*
grpcio_tools>=1.19.0
memory_profiler>=0.55
bitstring>=3.1.6
```

Finally, run pip to install the necessary python packages and set appropriate paths for the conda environment.
```
python -m pip install /PATH/TO/nxsdk-<version>/
conda develop /PATH/TO/nxsdk-<version>/nxsdk/
conda develop /PATH/TO/nxsdk-apps-<version>/nxsdk_modules/
```

# Fluence Miniapp

This miniapp approximates the fluence for a 2D particle transport problem. The miniapp's main purpose is to demonstrate some functions of the Neural Random Walker codebase.  It calculates the integrated-over-time particle flux for a one-dimensional particle traveling at various effective velocities.  The dimensions of the problem are space and direction.

It creates a transition matrix for particles transitioning among various (space, direction) coordinates and saves the matrix as $matrix_filename. Once the matrix is created, random walkers are simulated starting on every coordinate pair in the mesh.

There are several options, described below.  Some variables are required for the simulation to run.  These are M, dt, da, v, SigS, SigA, and L.  M is the number of walkers to start on each position in the mesh.  dt is the time discretization size.  da is the angular discritization size - this corresponds to the different 'direction' values the random walkers can assume.  v is the maximum velocity of the particle, achieved only when the particle has an angle of 1 (or -v when the particle has an angle of -1). SigS is the scattering cross-section (or rate) and SigA is the absorption cross-section (or rate).  L specifies the interval [-L,L] that the particle is allowed to exist in.

This problem has been adapted from the Nature Electronics paper here: https://www.nature.com/articles/s41928-021-00705-7. The default parameters have been altered to reduce the number of walkers simulated per location and to provide a more interesting source term.
```
  M = 1000
  dt = 0.01
  da = 0.0666667 (=1/15)
  v = 200
  SigS = 0.15
  SigA = 0
  L = 1
```
In addition, the Nature Electronics paper used a different source term. Specifically, S(x,w)=0.015 whenever |x|<0.5.  Here, x is position and w is angle.  The source term in this miniapp is coded as a separate function in the fluence_mini_app.py file.

Variables are taken as options from the command line and are parsed using argparse. The details and defaults of these variables are described below in the Usage section. Please note that balancing and optimizng the parameters is a challenging process. They must be selected so that the discrete time Markov chain well approximates the underling stochastic process and selected so that the lower precision probability on Loihi introduces negligible error.

## Usage
```
usage: fluence_mini_app.py [-h] [-r {fugu,loihi,spinnaker,pc}] [-M NUM_WALKS]
                           [-dt TIMESTEP] [-da ANGLESTEP] [-v VELOCITY]
                           [-ss SIGS] [-sa SIGA] [-L INTERVAL_LENGTH]
                           [-nt NEURAL_TIMESTEPS] [-d] [--verbose]
                           [--matrix_filename MATRIX_FILENAME]
                           [--results_filename RESULTS_FILENAME]
                           [--fugu_backend {snn}] [--plot_results]
                           [--log_file LOG_FILE] [--runclean RUNCLEAN]
                           [--rand_seed RAND_SEED] [--use_sinks]

optional arguments:
  -h, --help            show this help message and exit
  -r {fugu,loihi,spinnaker,pc}, --run_mode {fugu,loihi,spinnaker,pc}
                        The mode specifies how the random walks are ran. Must
                        be one of fugu, loihi, spinnaker, or pc. The default
                        is fugu. Specifying pc gives a python implementation.
  -M NUM_WALKS, --num_walks NUM_WALKS
                        The number of walkers to start on each location in the
                        mesh. The default is 100.
  -dt TIMESTEP, --timestep TIMESTEP
                        The time step of the simulation. The default is 0.01.
  -da ANGLESTEP, --anglestep ANGLESTEP
                        The angular discretization step size. The default is
                        1/15.
  -v VELOCITY, --velocity VELOCITY
                        The maximum velocity of the particle. The default is
                        200.
  -ss SIGS, --SigS SIGS
                        The scattering cross-section of the particle. This is
                        the rate at which scattering events occur. The default
                        is 0.15.
  -sa SIGA, --SigA SIGA
                        The absorption cross-section. This is the rate at
                        which particles are absorbed. The default is zero.
  -L INTERVAL_LENGTH, --interval_length INTERVAL_LENGTH
                        The length of the spatial interval [-L,L] the particle
                        is allowed to travel through. The default is 1.
  -nt NEURAL_TIMESTEPS, --neural_timesteps NEURAL_TIMESTEPS
                        The number of neural timesteps the simulation is run
                        for. This should be MUCH greater than the number of
                        model time steps desired. The default is 30000. For pc
                        run mode, this is the number of model timesteps.
  -d, --debug           Print logging DEBUG statements.
  --verbose             Print logging INFO statements.
  --matrix_filename MATRIX_FILENAME
                        Filename for temporary storage of transition matrix.
  --results_filename RESULTS_FILENAME
                        Filename to record the results.
  --fugu_backend {snn}  Runs a spiking simulator using Fugu.
  --plot_results        Plots the results after finishing.
  --log_file LOG_FILE   Location of the log file.
  --runclean RUNCLEAN   Run probe-free replicate in a Loihi run.
  --rand_seed RAND_SEED
                        Sets the seed for the random number generator.
                        Otherwise uses a time-based 'random' RNG seed.
  --use_sinks           Include sinks in a fugu simulation; default is False.
  ```

# Simple Flux Miniapp

This miniapp is a simplified particle flux problem. The miniapp's main purpose is to demonstrate some functions of the Neural Random Walker codebase. This miniapp will run much faster than the larger Fluence Miniapp. It calculates particle flux over time for a two state problem. Particles can be in one of two directions states: positive or negative.

It creates a transition matrix for the probability of a particle changing between the two states based on input variables and saves the matrix as $matrix_filename. Once the matrix is created, random walkers will be simulated starting on every position of the mesh. These will be simulated according to user specification.

There are several options, described below.  Some variables are required for the simulation to run.  These are M, dt, ip, in, SigS, and SigA.  M is the number of walkers to start on each position in the mesh.  dt is the time discretization size.  ip and in are the initial condition for the positive and negative solutions respectively. SigS is the scattering cross-section (or rate) and SigA is the absorption cross-section (or rate).

The problem approximates the fluence for two states in a two state problem over time.  It can be though of as providing a function F(t,1) and F(t,-1) where 1 is the positive direction state and -1 is the negative direction state. It is based off of SN3.8 from the Nature Electronics paper located here: https://www.nature.com/articles/s41928-021-00705-7
    
The default values in this simulation are different from the referenced paper. They have been selected to best approximate the solution with the Loihi hardware.

Variables are taken as options from the command line and are parsed using argparse. The details and defaults of these variables are described below in the Usage section. Please note that balancing and optimizng the parameters is a challenging process. They must be selected so that the discrete time Markov chain well approximates the underling stochastic process and selected so that the lower precision probability on Loihi introduces negligible error.

## Usage
```
usage: simple_flux_miniapp.py [-h] [-r {fugu,loihi,spinnaker,pc}]
                              [-M NUM_WALKS] [-dt TIMESTEP] [-ip INIT_POS]
                              [-in INIT_NEG] [-ss SIGS] [-sa SIGA]
                              [-mt MAX_TIME] [-nt NEURAL_TIMESTEPS] [--debug]
                              [--verbose] [--matrix_filename MATRIX_FILENAME]
                              [--results_filename RESULTS_FILENAME]
                              [--fugu_backend {snn}] [--plot_results]
                              [--log_file LOG_FILE] [--rand_seed RAND_SEED]
                              [--use_sinks]

optional arguments:
  -h, --help            show this help message and exit
  -r {fugu,loihi,spinnaker,pc}, --run_mode {fugu,loihi,spinnaker,pc}
                        The mode specifies how the random walks are ran. Must
                        be one of fugu, loihi, spinnaker, or pc. The default
                        is fugu. Specifying pc gives a python implementation.
  -M NUM_WALKS, --num_walks NUM_WALKS
                        The number of walkers to start on each location in the
                        mesh. The default is 500.
  -dt TIMESTEP, --timestep TIMESTEP
                        The time step of the simulation. The default is 0.005.
  -ip INIT_POS, --init_pos INIT_POS
                        The initial value for the positive direction solution.
                        The default is 5.
  -in INIT_NEG, --init_neg INIT_NEG
                        The initial value for the negative direction solution.
                        The default is 3.
  -ss SIGS, --SigS SIGS
                        The scattering cross-section of the particle. This is
                        the rate at which scattering events occur. The default
                        is 8.0.
  -sa SIGA, --SigA SIGA
                        The absorption cross-section. This is the rate at
                        which particles are absorbed. The default is 2.0.
  -mt MAX_TIME, --max_time MAX_TIME
                        The absolute max model time of simulation. This is the
                        largest value of time that could be displayed in the
                        solution. The actual largest model time of simulation
                        may be smaller than this value if the neural timestep
                        is too small for Loihi or Fugu->Loihi simulations. For
                        conventional (pc) simulations, the actual time may be
                        smaller if dt*nt<mt.
  -nt NEURAL_TIMESTEPS, --neural_timesteps NEURAL_TIMESTEPS
                        The number of neural timesteps the simulation is run
                        for. This should be MUCH greater than the number of
                        model time steps desired. The default is 500000. For
                        pc, this is the number of model timesteps.
  --debug               Print logging DEBUG statements.
  --verbose             Print logging INFO statements.
  --matrix_filename MATRIX_FILENAME
                        Filename for temporary storage of transition matrix.
  --results_filename RESULTS_FILENAME
                        Filename to record the results.
  --fugu_backend {snn}  Runs a spiking simulator using Fugu.
  --plot_results        Plots the results after finishing.
  --log_file LOG_FILE   Location of the log file.
  --rand_seed RAND_SEED
                        Sets the seed for the random number generator.
                        Otherwise uses a time-based 'random' RNG seed.
  --use_sinks           Include sinks in a fugu simulation; default is False.
  ```