import numpy as np
from tensorflow import keras
from train import createTrainingInstance, predictShot
import pandas as pd
from scipy.stats import ttest_ind
from tqdm import tqdm

def visualizeMatrix(m):
  output = '╔═════ BATTLESHIP ═════╗'
  chars = [*'⠀░▒▓█']
  percentiles = np.percentile(m, [25, 50, 75, 100])

  for c, x in np.ndenumerate(m):
    if c[1] == 0:
      output += '\n║ '
      
    if x <= percentiles[0]:
      output += '{0}{0}'.format(chars[0])
    elif x > percentiles[0] and x <= percentiles[1]:
      output += '{0}{0}'.format(chars[1])
    elif x > percentiles[1] and x <= percentiles[2]:
      output += '{0}{0}'.format(chars[2])
    elif x > percentiles[2] and x <= percentiles[3]:
      output += '{0}{0}'.format(chars[3])
    else:
      output += '{0}{0}'.format(chars[4])

    if c[1] == len(m[0]) - 1:
      output += ' ║'
  return output + '\n╚══════════════════════╝'

model = keras.models.load_model('trained_model')

def mlPlay(n):
  test = createTrainingInstance(0.0)

  hist = test[0]
  targets = test[1]
  for x in range(n):
    x, y = predictShot(hist, model)
    # feed results back into game state
    isHit = targets[x][y]
    if (isHit == 0): hist[x][y] = -1
    if (isHit == 1): hist[x][y] = 1

  # print match stats 
  newGameState = pd.DataFrame(hist).stack().reset_index()
  newGameState.columns = ['x', 'y', 'shotResult']
  hits = newGameState[newGameState['shotResult'] == 1]['shotResult'].sum()
  misses = newGameState[newGameState['shotResult'] == -1]['shotResult'].sum()
  return (hits, np.absolute(misses))

def randPlay(n):
  test = createTrainingInstance(0.0)

  hist = test[0]
  targets = test[1]
  for x in range(n):
    histDf = pd.DataFrame(hist).stack().reset_index()
    histDf.columns = ['x', 'y', 'shotResult']
    # compile all unknowns ie where we haven't taken a shot at
    unknowns = histDf[histDf['shotResult'] == 0]
    # get a random row
    p = unknowns.sample()
    x = int(p['x'])
    y = int(p['y'])
    # feed results back into game state
    isHit = targets[x][y]
    if (isHit == 0): hist[x][y] = -1
    if (isHit == 1): hist[x][y] = 1


  # print match stats 
  newGameState = pd.DataFrame(hist).stack().reset_index()
  newGameState.columns = ['x', 'y', 'shotResult']
  hits = newGameState[newGameState['shotResult'] == 1]['shotResult'].sum()
  misses = newGameState[newGameState['shotResult'] == -1]['shotResult'].sum()
  return (hits, np.absolute(misses))


histCols = ['hits', 'misses']

nGames = 50
nSteps = 50 

randHist = pd.DataFrame([], columns=histCols)
mlHist = pd.DataFrame([], columns=histCols)

progress = tqdm(range(nGames))

for g in range(nGames):
  progress.update()

  randHits, randMisses = randPlay(nSteps)
  newRandRow = pd.DataFrame([[randHits, randMisses]], columns=histCols)
  randHist = pd.concat([randHist, newRandRow])

  mlHits, mlMisses = mlPlay(nSteps)
  newMlRow = pd.DataFrame([[mlHits, mlMisses]], columns=histCols)
  mlHist = pd.concat([mlHist, newMlRow])
progress.close()

print('\n== RAND STATS ==')
print(randHist.describe(percentiles=[0.25, 0.5, 0.75]))
print('== MODEL STATS ==')
print(mlHist.describe(percentiles=[0.25, 0.5, 0.75]))

stat, pVal = ttest_ind(randHist['hits'], mlHist['hits'])
print('\nt-test results: {}, {}%'.format(
  np.round(stat, decimals = 2), 
  np.round(pVal, decimals = 2) * 100)
)