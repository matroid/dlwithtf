import numpy as np
from fcnet_func import eval_tox21_hyperparams

scores = {}
n_reps = 3
hidden_sizes = [5, 30, 60]
epochs = [5, 15, 30]

for rep in range(n_reps):
  for n_epochs in epochs:
    for hidden_size in hidden_sizes:
      score = eval_tox21_hyperparams(n_hidden=hidden_size, n_epochs=n_epochs)
      if (hidden_size, n_epochs) not in scores:
        scores[(hidden_size, n_epochs)] = []
      scores[(hidden_size, n_epochs)].append(score)
print("All Scores")
print(scores)

avg_scores = {}
for params, param_scores in scores.iteritems():
  avg_scores[params] = np.mean(np.array(param_scores))
print("Scores Averaged over repetitions")
print(avg_scores)
