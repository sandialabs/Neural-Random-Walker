# Copyright 2023 National Technology & Engineering Solutions of Sandia, LLC (NTESS). 
# Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains 
# certain rights in this software.
# Apache License Version 2.0, January 2004 http://www.apache.org/licenses/
import numpy as np
import argparse
from .DensityRW import load_transitions, DensityRW
from fugu.scaffold import Scaffold
from fugu.bricks import Vector_Input
import logging

rw_logger = logging.getLogger("NRW")

def init_walker_tuple(s):
  try:
    node_number, walkers = map(int, s.split(','))
    return (node_number, walkers)
  except:
    raise argparse.ArgumentTypeError("Must be (node_number, walkers) pair")


def run_miniapp(transitions_file=None,
                neural_timesteps = 100,
                initial_walkers = [],
                sink_connections=False,
                backend='snn'):
  transitions = load_transitions(transitions_file,
                                 remove_sink_connections = not(sink_connections),
                                 verbose=0)
  scaffold = Scaffold()
  if backend == 'snn':
    from fugu.backends import snn_Backend
    backend_object = snn_Backend()
  elif backend == 'loihi':
    from fugu.backends import loihi_Backend
    backend_object = loihi_Backend()
  else:
    raise ValueError("Unsupported backend type.")
  initial_walkers = { (t[0],):t[1] for t in initial_walkers}
  scaffold.add_brick(Vector_Input(np.array([1]),coding='Raster',name='Input0'),'input')
  scaffold.add_brick(DensityRW(neural_timesteps, 
                               transitions,
                               init_walkers = initial_walkers,
                               name = 'DensityRW'),
                    [0],
                    output=True)
  scaffold.lay_bricks()
  
  neuron_number_map = dict()
  rw_neurons = [node for node in scaffold.graph.nodes if ('groups' in scaffold.graph.nodes[node])]
  readout_neurons = [node for node in rw_neurons if ('readout' in scaffold.graph.nodes[node]['groups'])]
  for neuron in readout_neurons:
    neuron_number_map[scaffold.graph.nodes[neuron]['neuron_number']] = scaffold.graph.nodes[neuron]['groups'][1]
  backend_object.compile(scaffold)
  result = backend_object.run(neural_timesteps)
  node_col = []
  for neuron in result['neuron_number'].astype(int):
    node_col.append(neuron_number_map[neuron])
  result['node'] = node_col
  return result


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--transitions_file',required=True)
    parser.add_argument('--neural_timesteps',type=int,default=100)
    parser.add_argument('--remove_sink_connections',default=False,action='store_false',dest='sink_connections')
    parser.add_argument('--leave_sink_connections',action='store_true',dest='sink_connections')
    parser.add_argument('--initial_walkers',type=init_walker_tuple, nargs='*')
    parser.add_argument('--backend',choices=['snn','loihi'],default='snn')
    args = vars(parser.parse_args())
    rw_logger.info("Using parameters: {}".format(args))
    result = run_miniapp(**args)
    rw_logger.info(result)
