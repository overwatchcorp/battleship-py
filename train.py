import gym
import gym_battleship
import pandas as pd
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
import tqdm

boardDimensions = (10, 10)
totalCells = boardDimensions[0] * boardDimensions[1]

# create model

# generate training data
env = gym.make('Battleship-v0', board_size=boardDimensions)

def createTrainingInstance(noiseThreshold = 0.5):
  # create a valid board
  env.reset()
  shipLocations = env.board_generated
  # create a random matrix of "historical" data
  seeds = np.random.rand(totalCells)
  genHistVals = np.zeros(totalCells)
  for i, s in enumerate(seeds):
    if s < noiseThreshold:
      genHistVals[i] = -1
  # reshape into 2D array that matches board dimensions
  genHistVals = np.reshape(genHistVals, boardDimensions)
  # reference the ship locations to get hits and set value to 1
  for x, _ in enumerate(shipLocations):
    for y, s in enumerate(shipLocations[x]):
      if s == 1 and genHistVals[x, y] == -1:
        genHistVals[x, y] = 1
  return [genHistVals, shipLocations]

# make N instances of training data
def createTrainingSet(n = 1000):
  trainingSet = []
  for _ in range(n):
    # create training set of X
    data = createTrainingInstance(0)
    trainingSet.append(data)
  return trainingSet

# make a model 
def createModel():
  inputs = keras.Input(shape=boardDimensions)
  h1 = keras.layers.Dense(256, activation="relu", input_shape=(1, 100), name="h1")(inputs)
  h2 = keras.layers.Dense(512, activation="relu", input_shape=(1, 100), name="h2")(h1)
  outputs = layers.Dense(10)(h2)

  model = keras.Model(inputs, outputs, name="battleship_midgame_predictor")

  model.compile(
    loss=keras.losses.BinaryCrossentropy(),
    optimizer=keras.optimizers.Adam(),
    metrics=["accuracy"]
  )

  return model

# use the model to take a shot
def predictShot(inputState, inputModel):
  # data comes in as a 2D numpy matrix
  predMatrix = inputModel(inputState)
  # # print a nice vis for fun
  # print(visualizeMatrix(predMatrix))
  # convert to pandas dataframe, transform matrix into list w/ coords and value
  pred = pd.DataFrame(predMatrix).stack().reset_index()
  gameState = pd.DataFrame(inputState).stack().reset_index()
  # assign column names
  pred.columns = ['x', 'y', 'shipProbability']
  gameState.columns = ['x', 'y', 'shotResult']
  # merge into one frame
  merged = pred.merge(gameState, how='outer')
  # drop all the known squares and sort by shipProbability
  unknowns = merged[merged['shotResult'] == 0].sort_values(
    by="shipProbability", ascending=False)
  # get the best candidate for the next shot
  x, y, prob, _ = unknowns.iloc[0]
  x = int(x)
  y = int(y)
  return (x, y)


nGames = 100
nSteps = 30
# define training epoch amount
epochs = 10
# create empty games
trainingSet = createTrainingSet(nGames)


class trainingLogger(keras.callbacks.Callback):
  progressBar = tqdm.tqdm(range(nGames * nSteps))
  progressState = 0

  def on_train_begin(self, logs=None):
    self.progressState += 1
    self.progressBar.update(self.progressState)
  

def train():
  model = createModel()
  for gameInstance in trainingSet:
    gameState = gameInstance[0]
    targets = gameInstance[1]

    for _ in range(nSteps):
      model.fit(
        gameState,
        targets,
        epochs=epochs,
        verbose=0,
        callbacks=[trainingLogger()]
      )
      x, y = predictShot(gameState, model)
      # feed results back into game state
      isHit = targets[x][y]
      if (isHit == 0): gameState[x][y] = -1
      if (isHit == 1): gameState[x][y] = 1

  model.save('trained_model')