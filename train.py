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
def createTrainingSet(n = 1000, gameCompletion=np.random.rand()):
  trainingSet = []
  for _ in range(n):
    # create training set of X
    data = createTrainingInstance(gameCompletion)
    trainingSet.append(data)
  return trainingSet

# make a model 
def createModel():
  inputs = keras.Input(shape=boardDimensions)
  h1 = keras.layers.Dense(1024, activation="relu", input_shape=(1, 100), name="h1")(inputs)
  d1 = keras.layers.Dropout(0.2, name="d1")(h1)
  h2 = keras.layers.Dense(1024, activation="relu", input_shape=(1, 100), name="h2")(d1)
  d2 = keras.layers.Dropout(0.2, name="d2")(h2)
  outputs = layers.Dense(10)(d2)

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
  predList =  np.array(predMatrix).tolist()
  return (x, y, predList)


# define number of games to train on
nGames = 1000
nEndgames = 500
totalGames = nGames + nEndgames
# define training epoch amount
epochs = 10

class trainingLogger(keras.callbacks.Callback):
  progressBar = tqdm.tqdm(range(totalGames))
  progressState = 0

  def on_train_end(self, logs=None):
    self.progressState += 1
    # self.progressBar.write('loss: {}, accuracy: {}'.format(
      # np.round(logs['loss'], decimals=2), np.round(logs['accuracy'], decimals=2)))
    self.progressBar.update(self.progressState)
  

def train():
  model = createModel()
  # create empty games
  mainTrainingSet = createTrainingSet(nGames)
  endgameTrainingSet = createTrainingSet(nEndgames, 0.9)
  # concat regular games and endgames and then shuffle together
  trainingSet = np.concatenate((mainTrainingSet, endgameTrainingSet))
  np.random.shuffle(trainingSet)

  for gameInstance in trainingSet:
    gameState = gameInstance[0]
    targets = gameInstance[1]

    model.fit(
      gameState,
      targets,
      epochs=epochs,
      verbose=0,
      callbacks=[trainingLogger()]
    )

  return model
