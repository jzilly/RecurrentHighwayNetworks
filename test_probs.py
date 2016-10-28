"""A simple script to load and average MC testing predictions"""
import numpy as np
import os

sum_p = None
i = 0
count = 0

for i, probfile in enumerate(os.listdir('probs')):
  prob = np.load('probs/' + probfile)
  perp = np.exp(np.mean(-np.log(np.clip(prob, 1e-10, 1 - 1e-10))))
  if perp > 500:
    continue
  if i == 0:
    sum_p = prob
  else:
    sum_p += prob
  count += 1
  mean = np.exp(np.mean(-np.log(np.clip(sum_p/count, 1e-10, 1-1e-10)))) if count > 0 else 0
  print(count, probfile, perp, mean)
