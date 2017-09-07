"""Adapted from DeepChem Examples by Peter Eastman and Karl Leswing."""

import copy
import random
import shutil
import numpy as np
import tensorflow as tf
import deepchem as dc
from environment import TicTacToeEnvironment
#from a3c import Layer
#from a3c import Adam
#from a3c import Dense
#from a3c import Squeeze
#from a3c import BatchNorm
#from a3c import Flatten
#from a3c import SoftMax
from a3c import A3C


#class TicTacToePolicy(dc.rl.Policy):
#  def create_layers(self, state, **kwargs):
#    d1 = Flatten(in_layers=state)
#    d2 = Dense(
#        in_layers=[d1],
#        activation_fn=tf.nn.relu,
#        normalizer_fn=tf.nn.l2_normalize,
#        normalizer_params={"dim": 1},
#        out_channels=64)
#    d3 = Dense(
#        in_layers=[d2],
#        activation_fn=tf.nn.relu,
#        normalizer_fn=tf.nn.l2_normalize,
#        normalizer_params={"dim": 1},
#        out_channels=32)
#    d4 = Dense(
#        in_layers=[d3],
#        activation_fn=tf.nn.relu,
#        normalizer_fn=tf.nn.l2_normalize,
#        normalizer_params={"dim": 1},
#        out_channels=16)
#    d4 = BatchNorm(in_layers=[d4])
#    d5 = Dense(in_layers=[d4], activation_fn=None, out_channels=9)
#    value = Dense(in_layers=[d4], activation_fn=None, out_channels=1)
#    value = Squeeze(squeeze_dims=1, in_layers=[value])
#    probs = SoftMax(in_layers=[d5])
#    return {'action_prob': probs, 'value': value}


def eval_tic_tac_toe(value_weight,
                     num_epoch_rounds=1,
                     games=10**4,
                     rollouts=10**5,
                     advantage_lambda=0.98):
  """
  Returns the average reward over 10k games after 100k rollouts
  
  Parameters
  ----------
  value_weight: float

  Returns
  ------- 
  avg_rewards
  """
  env = TicTacToeEnvironment()
  model_dir = "/tmp/tictactoe"
  try:
    shutil.rmtree(model_dir)
  except:
    pass

  avg_rewards = []
  for j in range(num_epoch_rounds):
    print("Epoch round: %d" % j)
    a3c_engine = A3C(
        env,
        entropy_weight=0.01,
        value_weight=value_weight,
        model_dir=model_dir,
        advantage_lambda=advantage_lambda)
    try:
      a3c_engine.restore()
    except:
      print("unable to restore")
      pass
    a3c_engine.fit(rollouts)
    rewards = []
    for i in range(games):
      env.reset()
      reward = -float('inf')
      while not env.terminated:
        action = a3c_engine.select_action(env.state)
        reward = env.step(action)
      rewards.append(reward)
    print("Mean reward at round %d is %f" % (j+1, np.mean(rewards)))
    avg_rewards.append({(j + 1) * rollouts: np.mean(rewards)})
  return avg_rewards


def main():
  value_weight = 6.0
  score = eval_tic_tac_toe(value_weight=0.2, num_epoch_rounds=20,
                           advantage_lambda=0.,
                           games=10**4, rollouts=5*10**4)
  print(score)


if __name__ == "__main__":
  main()
