import gym
import gym_battleship
import pandas as pd
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
from tqdm.keras import TqdmCallback


boardDimensions = (10, 10)
totalCells = boardDimensions[0] * boardDimensions[1]

# create model

# generate training data
def createTrainingInstance(noiseThreshold = 0.5):
  # create a valid board
  env = gym.make('Battleship-v0', board_size=boardDimensions)
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
  # set all the ships locations as so-far untouched
  # for x, _ in enumerate(shipLocations):
  #   for y, s in enumerate(shipLocations[x]):
  #     if s == 1:
  #       genHistVals[x, y] = 1
  return [genHistVals, shipLocations]

# make N instances of training data
def createTrainingSet(n = 1000):
  trainingSet = []
  for _ in range(n):
    # create training set of X
    data = createTrainingInstance(np.random.rand())
    trainingSet.append(data)
  return trainingSet

# make a model 
def createModel():
  inputs = keras.Input(shape=boardDimensions)
  h1 = keras.layers.Dense(256, activation="relu", input_shape=(1, 100), name="h1")(inputs)
  h2 = keras.layers.Dense(512, activation="relu", input_shape=(1, 100), name="h2")(inputs)
  h3 = keras.layers.Dense(256, activation="relu", input_shape=(1, 100), name="h3")(inputs)
  outputs = layers.Dense(10)(h3)

  model = keras.Model(inputs, outputs, name="battleship_midgame_predictor")

  model.compile(
    loss=keras.losses.BinaryCrossentropy(),
    optimizer=keras.optimizers.Adam(),
    metrics=["accuracy"]
  )

  return model

trainingSet = createTrainingSet(10000)

def train():
  model = createModel()
  for gameInstance in trainingSet:
    gameState = gameInstance[0]
    targets = gameInstance[1]

    epochs = 10

    model.fit(
      gameState,
      targets,
      epochs=epochs,
      verbose=0,
      callbacks=[TqdmCallback(verbose=0, epochs=epochs)]
    )
  model.save('trained_model')