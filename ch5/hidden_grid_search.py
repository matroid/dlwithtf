import numpy as np
from fcnet_func import eval_tox21_hyperparams

scores = {}
n_reps = 3
hidden_sizes = [30, 60]
epochs = [15, 30, 45]
dropouts = [.5]
num_layers = [1, 2]

for rep in range(n_reps):
  for n_epochs in epochs:
    for hidden_size in hidden_sizes:
      for dropout in dropouts:
        for n_layers in num_layers:
          score = eval_tox21_hyperparams(n_hidden=hidden_size, n_epochs=n_epochs,
                                         dropout_prob=dropout, n_layers=n_layers)
          if (hidden_size, n_epochs, dropout, n_layers) not in scores:
            scores[(hidden_size, n_epochs, dropout, n_layers)] = []
          scores[(hidden_size, n_epochs, dropout, n_layers)].append(score)
print("All Scores")
print(scores)

avg_scores = {}
for params, param_scores in scores.iteritems():
  avg_scores[params] = np.mean(np.array(param_scores))
print("Scores Averaged over %d repetitions" % n_reps)
print(avg_scores)
