import numpy as np
import matplotlib.pyplot as plt

from util import getData

label_maps = ['Anger', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']


def main():
  X, Y = getData(balance_ones=False)
  while True:
    for i in range(len(label_maps)):
      x, y = X[Y == i], Y[Y == i]
      N = len(y)
      j = np.random.choice(N)
      plt.imshow(x[j].reshape(48, 48), cmap='gray')
      plt.title(label_maps[y[j]])
      plt.show()
    prompt = input('Quit?')
    if prompt == 'y':
      break


if __name__ == '__main__':
  main()
