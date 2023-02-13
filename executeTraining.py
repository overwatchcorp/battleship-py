from train import train

model = train()
print('training complete! saving model to ./trained_model')
model.save('trained_model')