import numpy as np


def getData(balance_ones=True):
  # images are 48x48 = 2304 size vectors
  Y = []
  X = []
  first = True
  for line in open('fer2013.csv'):
    if first:
      first = False
    else:
      row = line.split(',')
      Y.append(int(row[0]))
      X.append([int(p) for p in row[1].split()])

  X, Y = np.array(X) / 255.0, np.array(Y)

  if balance_ones:
    # balance the 1 class
    X0, Y0 = X[Y != 1, :], Y[Y != 1]
    X1 = X[Y == 1, :]
    X1 = np.repeat(X1, 9, axis=0)
    X = np.vstack([X0, X1])
    Y = np.concatenate((Y0, [1]*len(X1)))

  return X, Y


def getBinaryData():
  Y = []
  X = []
  first = True
  for line in open('fer2013.csv'):
    if first:
      first = False
    else:
      row = line.split(',')
      y = int(row[0])
      if y == 0 or y == 1:
        Y.append(y)
        X.append([int(p) for p in row[1].split()])
  return np.array(X) / 255.0, np.array(Y)


def sigmoid(A):
  return 1 / (1 + np.exp(-A))


def sigmoid_cost(T, Y):
  return -(T*np.log(Y) + (1-T)*np.log(1-Y)).sum()


def error_rate(targets, predictions):
  return np.mean(targets != predictions)
