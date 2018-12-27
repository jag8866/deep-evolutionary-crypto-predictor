'''
Original attempt at a neural net crypto price predictor. Uses a single neural network model with a set
group of exchange symbols, takes in a window of previous price information and attempts to predict the price
some number of steps ahead.
'''

from pandas import *
from keras import *
import matplotlib.pyplot as plt
import numpy
from numpy import newaxis
import math
import ccxt
from dataset import *
numpy.random.seed(7)

def predict_sequence_full(model, data, window_size):
    '''
    Shift the window by 1 new prediction each time, re-run predictions on new window
    '''
    curr_frame = data[0]
    predicted = []
    for i in range(len(data)):
        predicted.append(model.predict(curr_frame[newaxis,:,:])[0,0])
        curr_frame = curr_frame[1:]
        curr_frame = numpy.insert(curr_frame, [window_size-1], predicted[-1], axis=0)
    return predicted

def predict_sequences_multiple(model, data, window_size, prediction_len):
    '''
    Predict sequence of prediction_len steps before shifting prediction run forward by 50 steps
    '''
    prediction_seqs = []
    for i in range(int(len(data)/prediction_len)):
        curr_frame = data[i*prediction_len]
        predicted = []
        for j in range(prediction_len):
            predicted.append(model.predict(curr_frame[newaxis,:,:])[0,0])
            curr_frame = curr_frame[1:]
            curr_frame = numpy.insert(curr_frame, [window_size-1], predicted[-1], axis=0)
        prediction_seqs.append(predicted)
    return prediction_seqs

def plot_results(predicted_data, true_data):
    fig = plt.figure(facecolor='white')
    ax = fig.add_subplot(111)
    ax.plot(true_data, label='True Data')
    plt.plot(predicted_data, label='Prediction')
    plt.legend()
    plt.show()

def plot_results_multiple(predicted_data, true_data, prediction_len):
    fig = plt.figure(facecolor='white')
    ax = fig.add_subplot(111)
    ax.plot(true_data, label='True Data')
    #Pad the list of predictions to shift it in the graph to it's correct start
    for i, data in enumerate(predicted_data):
        padding = [None for p in range(i * prediction_len)]
        plt.plot(padding + data, label='Prediction')
        plt.legend()
    plt.show()


look_back=14
look_ahead=7
symbols = ['BTC/USDT', 'ETH/USDT', 'BNB/BTC']
X, Y, df = create_exchange_multiset(look_back, look_ahead, symbols=symbols)

train_size = int(len(X) * 0.9)
test_size = len(X) - train_size
trainX, testX = X[0:train_size], X[train_size:len(X)]
trainY, testY = Y[0:train_size], Y[train_size:len(Y)]

#Create and fit the neural net
model = Sequential()
#model.add(layers.Conv1D(15, 5, input_shape=(look_back, 1), activation='relu', kernel_initializer='glorot_uniform', bias_initializer='glorot_uniform'))
#model.add(layers.Dropout(.3))
#model.add(layers.BatchNormalization())
#model.add(layers.MaxPooling1D(4))
#model.add(layers.GaussianNoise(.3))
model.add(layers.LSTM(128, return_sequences=True, input_shape=(5 * len(symbols), look_back), activation='relu', kernel_initializer='glorot_uniform', bias_initializer='glorot_uniform'))
#model.add(layers.Dropout(.3))
model.add(layers.LSTM(64, activation='relu', kernel_initializer='glorot_uniform', bias_initializer='glorot_uniform'))
#model.add(layers.Dropout(.3))
#model.add(layers.Conv1D(84, 3, activation='relu', kernel_initializer='random_uniform', bias_initializer='random_uniform'))
#model.add(layers.Dropout(.3))
#model.add(layers.BatchNormalization())
#model.add(layers.MaxPooling1D(2))
model.add(layers.Dense(1, activation='tanh', kernel_initializer='glorot_uniform', bias_initializer='glorot_uniform'))
model.compile(loss='mean_squared_error', optimizer=optimizers.Nadam(lr=0.002))
#reduce_lr = callbacks.ReduceLROnPlateau(monitor='val_acc', factor=0.9, patience=30, min_lr=0.000001, verbose=1)
#history = model.fit(trainX, trainY, validation_split=0.2, epochs=50, batch_size=30, verbose=2, callbacks=[reduce_lr])
history = model.fit(trainX, trainY, validation_split=0.3, epochs=1000, batch_size=5, verbose=2)

#Plotting the training and validation stats to help detect overfitting
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(len(loss))
plt.figure()
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title("Training and Validation Loss")
plt.legend()
plt.show()


#Text outputs of estimates of model performance
trainScore = model.evaluate(trainX, trainY, verbose=0)
print('Train Score: %.2f MSE (%.2f RMSE)' % (trainScore, math.sqrt(trainScore)))
testScore = model.evaluate(testX, testY, verbose=0)
print('Test Score: %.2f MSE (%.2f RMSE)' % (testScore, math.sqrt(testScore)))


#Now for further visualization we want to make predictions and graph them alongside the actual data
trainPredict = model.predict(trainX)
testPredict = model.predict(testX)

normalized_dataset = ((((df-df.min())/(df.max()-df.min()) * 2) - 1))

#shift train predictions for plotting
trainPredictPlot = numpy.empty_like(normalized_dataset)
trainPredictPlot[:, :] = numpy.nan
trainPredictPlot[look_back:len(trainPredict)+look_back, :] = trainPredict

#shift test predictions for plotting
testPredictPlot = numpy.empty_like(normalized_dataset)
testPredictPlot[:, :] = numpy.nan
testPredictPlot[len(trainPredict)+(look_back*2)+look_ahead:len(normalized_dataset)-look_ahead, :] = testPredict

#Attempt to reconstruct (pcnt change) predictions onto the original graph. Only works with percent change
FinalTrainPlot = numpy.empty_like(normalized_dataset)
FinalTrainPlot[:, :] = numpy.nan
FinalTestPlot = numpy.empty_like(normalized_dataset)
FinalTestPlot[:, :] = numpy.nan

for i in range(len(dataset)):
	if i >= look_back + look_ahead and i < look_back + look_ahead + len(trainPredict):
		FinalTrainPlot[i, 0] = (dataset[i-look_ahead, 0] * numpy.arctanh(trainPredictPlot[i, 0])) + dataset[i-look_ahead, 0]
	elif i >= look_back + look_ahead + len(trainPredict):
		FinalTestPlot[i, 0] = (dataset[i - look_ahead, 0] * numpy.arctanh(testPredictPlot[i, 0])) + dataset[i - look_ahead, 0]
FinalTrainPlot = numpy_minmax_normalize(FinalTrainPlot, df.min(), df.max())
FinalTestPlot = numpy_minmax_normalize(FinalTestPlot, df.min(), df.max())


# plot baseline and predictions
plt.plot(normalized_dataset)
plt.plot(trainPredictPlot)
plt.plot(testPredictPlot)
plt.plot(FinalTrainPlot)
plt.plot(FinalTestPlot)
plt.show()
print("\n")
