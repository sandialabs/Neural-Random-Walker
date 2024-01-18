# Copyright 2023 National Technology & Engineering Solutions of Sandia, LLC (NTESS). 
# Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains 
# certain rights in this software.
# Apache License Version 2.0, January 2004 http://www.apache.org/licenses/
# %config IPCompleter.greedy=True
import nxsdk.api.n2a as nx
import networkx as networkx

import logging

from nxsdk.graph.monitor.probes import PerformanceProbeCondition
from nxsdk.api.enums.api_enums import ProbeParameter
from nxsdk.graph.monitor.probes import *

from .rw_core import *
from .rw_mesh import *

rw_logger = logging.getLogger("NRW")

def test_network_short(custom_network,
                       x,
                       y,
                       time,
                       core_density=10,
                       location=[(0,0)],
                       num_walkers=100,
                       track = 0,
                       verbose = 0,
                       input_times = [1]):
    superNet=nx.NxNet()
    tprobe1=[]
    eprobe1=[]
    rw_logger.info('Setting up network')
    spikeReceiver=nx.SpikeReceiver(superNet)

    network_output = makeNetwork(superNet=superNet,
                                 x=x,
                                 y=y,
                                 custom_network=custom_network,
                                 core_density=core_density,
                                 input_x=location,
                                 sim_time=time,
                                 num_walkers=num_walkers,
                                 track=track,
                                 verbose = verbose,
                                 input_times = input_times,
                                 s=spikeReceiver)
    core_neuroncount, max_coreid, track_probes, chip_count, spikeReceiver = network_output

    rw_logger.info('Compiling network into NxNet')
    compiler = nx.N2Compiler()
    board = compiler.compile(superNet)
    if track==2:
        probeCond1 = PerformanceProbeCondition(tStart=1, tEnd=time, bufferSize=1000, binSize=8)
        probeCond2 = PerformanceProbeCondition(tStart=1, tEnd=time, bufferSize=1000, binSize=8)
        tprobe1 = board.probe(ProbeParameter.EXECUTION_TIME, probeCond1)
        eprobe1 = board.probe(ProbeParameter.ENERGY, probeCond2)

    rw_logger.info("Running with {} cores on {} chips".format(max_coreid, chip_count) )

    board.run(time, aSync=True)
    i=0

    board.finishRun()
    board.disconnect()
    superNet.disconnect()
    del superNet
    return tprobe1, eprobe1, track_probes, core_neuroncount, max_coreid

def rad_test(custom_network,
             location,
             time = 1000,
             num_walkers=100,
             verbose = 0,
             core_density=10,
             input_times=[1],
             track=1,
             size = (901,1)):
    x = size[0]
    y = size[1]

    test_results = test_network_short(custom_network,
                                      x,
                                      y,
                                      time,
                                      core_density,
                                      location,
                                      num_walkers,
                                      track,
                                      verbose = verbose,
                                      input_times = input_times)
    tprobe1, eprobe1, track_probes, core_neuroncount, max_coreid = test_results
    return tprobe1, eprobe1, track_probes, core_neuroncount, max_coreid
