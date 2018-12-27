from pandas import *
import matplotlib.pyplot as plt
import numpy
from numpy import newaxis
import math
import ccxt
numpy.random.seed(7)

def minmax_normalize(n, min, max):
	return ((((n - min) / (max - min)) * 2) - 1)

def numpy_minmax_normalize(dataset, min, max):
	newset = numpy.empty_like(dataset)
	newset[:, :] = numpy.nan
	for i in range(len(dataset)):
		newset[i, 0] = minmax_normalize(dataset[i, 0], min, max)
	return newset

def sma(data, window):
	if len(data) < window:
		raise ValueError("data is too short")
	return sum(data[-window:]) / float(window)

def ema(data, window):
    if len(data) < 2 * window:
        return None
    c = 2.0 / (window + 1)
    current_ema = sma(data[-window*2:-window], window)
    for value in data[-window:]:
        current_ema = (c * value) + ((1 - c) * current_ema)
    return current_ema

'''
Creates a basic dataset - takes in data and organizes it into windows (X) and
the new values look_ahead steps ahead of the last spot in the window (Y)
'''
def create_dataset(dataset, look_back=1, look_ahead=1):
	dataX, dataY = [], []
	for i in range(len(dataset)-look_back-look_ahead):
		a = dataset[i:(i+look_back), 0]
		max = a.max()
		min = a.min()
		b = []
		for j in range(0, len(a)):
			b.append((((a[j] - min)/(max - min)) * 2) - 1)

		dataX.append(b)
		lookahead_rawval = dataset[i + look_back + look_ahead - 1, 0]
		lookahead_pcntchange = (lookahead_rawval - dataset[i+look_back][0])/dataset[i+look_back][0]
		dataY.append(numpy.tanh(lookahead_pcntchange))

	dataX = np.expand_dims(dataX, axis=2)
	return numpy.array(dataX), numpy.array(dataY)

'''
Advanced version of dataset that includes exponential moving averages
'''
def create_advanced_dataset(dataset, look_back=1, look_ahead=1):
	dataX, dataY = [], []
	for i in range(160, len(dataset)-look_back-look_ahead):
		a = dataset[i:(i+look_back), 0]
		max = a.max()
		min = a.min()
		b = []

		#ema_20 = []
		#ema_40 = []
		ema_80 = []
		for j in range(0, len(a)):
			#ema_20.append(ema(dataset[:(i + j), 0], 20))
			#ema_40.append(ema(dataset[:(i + j), 0], 40))
			ema_80.append(ema(dataset[:(i + j), 0], 80))

		#ema_20=numpy.array(ema_20)
		#ema_40 = numpy.array(ema_40)
		ema_80 = numpy.array(ema_80)

		#ema_20_nrml = []
		#ema_40_nrml = []
		ema_80_nrml = []
		for j in range(0, len(a)):
			#try:
				#ema_20_nrml.append(minmax_normalize(ema_20[j], ema_20.min(), ema_20.max()))
			#except:
				#ema_20_nrml.append(None)
			#try:
				#ema_40_nrml.append(minmax_normalize(ema_40[j], ema_40.min(), ema_40.max()))
			#except:
				#ema_40_nrml.append(None)
			try:
				ema_80_nrml.append(minmax_normalize(ema_80[j], ema_80.min(), ema_80.max()))
			except:
				ema_80_nrml.append(None)

		for j in range(0, len(a)):
			normal_price = (((a[j] - min)/(max - min)) * 2) - 1
			b.append([normal_price, ema_80_nrml[j]])

		dataX.append(b)

		lookahead_rawval = dataset[i + look_back + look_ahead - 1, 0]
		lookahead_pcntchange = (lookahead_rawval - dataset[i+look_back][0])/dataset[i+look_back][0]
		dataY.append(numpy.tanh(lookahead_pcntchange))
	return numpy.array(dataX), numpy.array(dataY)

'''
Far superior dataset function. Pulls data directly from binance (by default). Can have as many coins as you want;
the last coin in symbols will be used as the target (the one we are trying to predict).
'''
def create_exchange_multiset(look_back=1, look_ahead=1, exchange=ccxt.binance(), symbols=['BTC/USDT', 'ETH/USDT'], period = '12h', trainsplit = .75):
	if look_back == 0: raise Exception("look_back not set")
	exchange.load_markets()
	price_history = []
	for symbol in symbols:
		price_history += [exchange.fetch_ohlcv(symbol, period)]

	debug_datahits = [0, 0, 0]
	rawchart = numpy.array(price_history[-1])

	dataX, dataY = [], []

	for i in range(len(price_history[0]) - look_back - look_ahead):
		chunk = []
		for p in price_history:
			if len(p) != len(price_history[0]):
				raise Exception("Price history lengths do not match")

			pruned_prices = numpy.swapaxes(numpy.delete(numpy.array(p), 0, 1), 0, 1)

			c = []
			#Move through each list, in order: O,H,L,C,V
			for price_type in pruned_prices:
				a = price_type[i:(i + look_back)]
				max = a.max()
				min = a.min()
				b = []
				for j in range(0, len(a)):
					b.append((((a[j] - min) / (max - min)) * 2) - 1)
				c.append(b)

			chunk.extend(c)

		'''
		# Create buy/sell/hold signal Y values
		lookahead_rawval = price_history[-1][i + look_back + look_ahead - 1][1]
		lookahead_pcntchange = (lookahead_rawval - price_history[-1][i + look_back][1])/price_history[-1][i + look_back][1]
		if lookahead_pcntchange > .01:
			dataY.append([1, 0, 0])
			debug_datahits[0] += 1
		elif lookahead_pcntchange < -.01:
			dataY.append([0, 0, 1])
			debug_datahits[2] += 1
		else:
			dataY.append([0, 1, 0])
			debug_datahits[1] += 1
		'''

		#Create percent change Y values
		lookahead_rawval = price_history[-1][i + look_back + look_ahead - 1][1]
		lookahead_pcntchange = (lookahead_rawval - price_history[-1][i+look_back][1])/price_history[-1][i+look_back][1]
		dataY.append(lookahead_pcntchange)

		dataX.append(chunk)

	return numpy.array(dataX), numpy.array(dataY), rawchart



'''
Example of the data the data the exchange gives us
[
        1504541580000, // UTC timestamp in milliseconds, integer
        4235.4,        // (O)pen price, float
        4240.6,        // (H)ighest price, float
        4230.0,        // (L)owest price, float
        4230.7,        // (C)losing price, float
        37.72941911    // (V)olume (in terms of the base currency), float
    ],
'''

