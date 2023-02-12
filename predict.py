import numpy as np
from tensorflow import keras
from train import createTrainingInstance

def visualizeMatrix(m):
  output = ''
  chars = [*'⠀░▒▓█']
  percentiles = np.percentile(m, [25, 50, 75, 100])

  for c, x in np.ndenumerate(m):
    if c[1] == 0:
      output += '\n'
      
    if x == np.max(m):
      output += '██'
    elif x <= percentiles[0]:
      output += '{0}{0}'.format(chars[0])
    elif x > percentiles[0] and x <= percentiles[1]:
      output += '{0}{0}'.format(chars[1])
    elif x > percentiles[1] and x <= percentiles[2]:
      output += '{0}{0}'.format(chars[2])
    elif x > percentiles[2] and x <= percentiles[3]:
      output += '{0}{0}'.format(chars[3])
    else:
      output += '{0}{0}'.format(chars[4])
  return output

model = keras.models.load_model('trained_model')

test = createTrainingInstance(0)

hist = test[0]
targets = test[1]

# print(test)

for x in range(50):
  pred = model.predict(test[0])
  maxX = 0
  maxY = 0
  maxV = -1
  for c, v in np.ndenumerate(pred):
    y = c[1]
    x = c[0]
    if v > maxV:
      maxV = v
      maxX = x
      maxY = y
  vis = visualizeMatrix(pred)
  print('{}\n({},{})'.format(vis, maxX, maxY))
  res = targets[maxX][maxY]
  if (res == 0): hist[maxX][maxY] = -1
  else: hist[maxX][maxY] = 1

 


# print('{}\n\n{}'.format(test[0], test[1]))

# print('== INPUT =={}\n== PREDICTION =={}'.format(
#   visualizeMatrix(test[0]), 
#   visualizeMatrix(pred)
#   # visualizeMatrix(test[1]))
# ))
