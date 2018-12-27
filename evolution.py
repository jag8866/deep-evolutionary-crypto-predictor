'''
Genetic algorithm to assist in finding network structures and symbol combinations that are ideal for predicting future
prices. Since layers are extremely picky about the shapes of their tensor inputs it allows networks with incompatible
layers to be created and simply kills them off when they fail to compile, thus selecting for properly constructed nets.
'''
import pprint
import random
from keras import *
import matplotlib.pyplot as plt
import numpy
import math
from dataset import *
from name_generator import random_name
import gc
import pickle
random.seed(7)
numpy.random.seed(7)

class Network():
    '''
    Class that represents the network to be evolved.
    '''
    def __init__(self):
        self.name = ""
        self.fitness  = 0
        self.period = '12h'
        self.look_back = 0
        self.look_ahead = 0
        self.symbols = []
        self.layers = [("Dense", "tanh", 1, .25, False)]
        self.network = Sequential()
        self.loss = 'mean_squared_error'
        self.optimizer = 'Nadam'
        self.train_plot = None
        self.trained = False
        self.train_epochs = 50


    def randomize(self):
        '''
        Completely randomize the attributes of the network
        '''
        self.period = random.choice(['1m', '30m', '1h', '2h', '6h', '12h'])
        self.look_back = random.randrange(15, 60)
        self.look_ahead = random.randrange(1, 15)
        self.loss = random.choice(["mean_squared_error", "mean_absolute_error", "mean_squared_logarithmic_error",
                                   "squared_hinge", "sparse_categorical_crossentropy", "kullback_leibler_divergence",
                                   "cosine_proximity"])
        self.optimizer = random.choice([optimizers.Nadam(), optimizers.SGD(), optimizers.RMSprop(), optimizers.Adagrad()])

        #Generate layers
        depth = random.randrange(1, 8)
        self.layers = [None] * depth
        for l in range(depth):
            if l ==0:
                self.layers[l] = self.randomLayer(top=True)
            else:
                self.layers[l] = self.randomLayer()

        self.train_epochs = random.randrange(50, 200)

        #Generate symbols list
        self.symbols = self.randomSymbolSet()

    def build(self, input):
        '''
        Build the network - we take our list of layer attributes and use them to make actual layers in the model.
        '''
        self.network = Sequential()

        #Input layer
        if self.layers[0][0] == "Dense":
            if self.layers[0][4]:
                self.network.add(layers.TimeDistributed(layers.Dense(self.layers[0][2], activation=self.layers[0][1], input_shape=input.shape)))
            else:
                self.network.add(layers.Dense(self.layers[0][2], activation=self.layers[0][1], input_shape=input.shape))
        if self.layers[0][0] == "Conv":
            if self.layers[0][4]:
                self.network.add(layers.TimeDistributed(layers.Conv1D(self.layers[0][2], 3, activation=self.layers[0][1], input_shape=input.shape)))
            else:
                self.network.add(layers.Conv1D(self.layers[0][2], 3, activation=self.layers[0][1], input_shape=input.shape))
        if self.layers[0][0] == "LSTM":
            if self.layers[0][4]:
                self.network.add(layers.LSTM(self.layers[0][2], activation=self.layers[0][1], input_shape=input.shape, return_sequences=True))
            else:
                self.network.add(layers.LSTM(self.layers[0][2], activation=self.layers[0][1], input_shape=input.shape))
        if self.layers[0][0] == "Conv+Pool":
            if self.layers[0][4]:
                self.network.add(layers.TimeDistributed(layers.Conv1D(self.layers[0][2], 3, activation=self.layers[0][1], input_shape=input.shape)))
                self.network.add(layers.TimeDistributed(layers.MaxPooling1D(int(numpy.round_(self.layers[0][3] * 10)))))
            else:
                self.network.add(layers.Conv1D(self.layers[0][2], 3, activation=self.layers[0][1], input_shape=input.shape))
                self.network.add(layers.MaxPooling1D(int(numpy.round_(self.layers[0][3] * 10))))
        if self.layers[0][0] == "Reshape":
            #We will use TimeDist to switch between these two types of reshapes
            if self.layers[0][4]:
                self.network.add(layers.Reshape(numpy.expand_dims(input.shape, axis=1).shape))
            else:
                self.network.add(layers.Reshape((-1, 1)))

        #Successive layers
        for i in range(1, len(self.layers)):
            if self.layers[i][0] == "Dense":
                if self.layers[i][4]:
                    self.network.add(layers.TimeDistributed(layers.Dense(self.layers[i][2], activation=self.layers[i][1])))
                else:
                    self.network.add(layers.Dense(self.layers[i][2], activation=self.layers[i][1]))
            if self.layers[i][0] == "Conv":
                #self.network.add(layers.Reshape((-1,1)))
                if self.layers[i][4]:
                    self.network.add(
                        layers.TimeDistributed(layers.Conv1D(self.layers[i][2], 3, activation=self.layers[i][1])))
                else:
                    self.network.add(layers.Conv1D(self.layers[i][2], 3, activation=self.layers[i][1]))
            if self.layers[i][0] == "LSTM":
                #self.network.add(layers.Reshape((-1, 1)))
                if self.layers[i][4]:
                    self.network.add(layers.LSTM(self.layers[i][2], activation=self.layers[i][1], return_sequences=True))
                else:
                    self.network.add(layers.LSTM(self.layers[i][2], activation=self.layers[i][1]))
            if self.layers[i][0] == "GausNoise":
                if self.layers[i][4]:
                    self.network.add(layers.TimeDistributed(layers.GaussianNoise(self.layers[i][3])))
                else:
                    self.network.add(layers.GaussianNoise(self.layers[i][3]))
            if self.layers[i][0] == "Dropout":
                if self.layers[i][4]:
                    self.network.add(layers.TimeDistributed(layers.Dropout(self.layers[i][3])))
                else:
                    self.network.add(layers.Dropout(self.layers[i][3]))
            if self.layers[i][0] == "BatchNorm":
                if self.layers[i][4]:
                    self.network.add(layers.TimeDistributed(layers.BatchNormalization()))
                else:
                    self.network.add(layers.BatchNormalization())
            if self.layers[i][0] == "Conv+Pool":
                #self.network.add(layers.Reshape((-1, 1)))
                if self.layers[i][4]:
                    self.network.add(layers.TimeDistributed(layers.Conv1D(self.layers[i][2], 3, activation=self.layers[i][1])))
                    self.network.add(layers.TimeDistributed(layers.MaxPooling1D(pool_size=numpy.round_(self.layers[i][3] * 10))))
                else:
                    self.network.add(layers.Conv1D(self.layers[i][2], 3, activation=self.layers[i][1]))
                    self.network.add(layers.MaxPooling1D(numpy.round_(self.layers[i][3] * 10)))
            if self.layers[i][0] == "Reshape":
                # We will use TimeDist to switch between these two types of reshapes
                if self.layers[i][4]:
                    self.network.add(layers.Reshape(numpy.expand_dims(input_shape, axis=1).shape))
                else:
                    self.network.add(layers.Reshape((-1, 1)))
            if self.layers[i][0] == "Flatten":
                # We will use TimeDist to switch between these two types of reshapes
                if self.layers[i][4]:
                    self.network.add(layers.Flatten())
                else:
                    self.network.add(layers.TimeDistributed(layers.Flatten()))
        #self.network.add(layers.Flatten())
        self.network.add(layers.Dense(1, activation='tanh'))
        self.network.compile(loss=self.loss, optimizer=self.optimizer)
        return input


    def train(self, verbose=False, stop_for_errors=False, quicktrain=False):
        """
        Train the network and record the fitness score.
        """
        trainX, trainY, df = create_exchange_multiset(look_back=self.look_back, look_ahead=self.look_ahead, symbols=self.symbols)

        if verbose:
            print("Now attempting to build and train the following network: ")
            print("Name: " + self.name)
            print("Symbols: " + str(self.symbols))
            print("Input shape: " + str(trainX.shape))
            pprint.pprint(self.layers)

        try:
            trainX = self.build(trainX)

            if quicktrain: epochs = 5
            else:
                epochs = self.train_epochs
            history = self.network.fit(trainX, trainY, validation_split=0.3, epochs=epochs, batch_size=10, verbose=2)
            loss = history.history['loss']
            val_loss = history.history['val_loss']
            epochs = range(len(loss))

            avg1 = numpy.mean([val_loss[(len(val_loss) // 2)], val_loss[(len(val_loss) // 2) - 1], val_loss[
                (len(val_loss) // 2) - 2], val_loss[(len(val_loss) // 2) - 3]])
            avg2 = numpy.mean([val_loss[0], val_loss[1], val_loss[2], val_loss[3]])
            early_pct_change = ((avg1 - avg2 / avg2) * 100)

            avg1 = numpy.mean([val_loss[len(val_loss)-1], val_loss[len(val_loss)-2], val_loss[len(val_loss)-3], val_loss[len(val_loss)-4]])
            avg2 = numpy.mean([val_loss[(len(val_loss) // 2)], val_loss[(len(val_loss) // 2)-1], val_loss[(len(val_loss) // 2)-2], val_loss[(len(val_loss) // 2)-3]])
            late_pct_change = (avg1 - avg2 / avg2) * 100

            self.fitness = late_pct_change + (early_pct_change / 3)
            if math.isnan(self.fitness):
                self.fitness = -1

            #Graph and output
            plt.figure()
            plt.plot(epochs, loss, 'bo', label='Training loss')
            plt.plot(epochs, val_loss, 'b', label='Validation loss')
            plt.title("Training and Validation Loss " + self.name + " - Fitness: " + str(self.fitness))
            plt.legend()
            self.train_plot = plt
            print("====================== Netword added to population! ======================")
            print("Name: " + self.name)
            print("Symbols: " + str(self.symbols))
            print("Input shape: " + str(trainX.shape))
            pprint.pprint(self.layers)

        except Exception as e:
            if stop_for_errors:
                raise e
            else:
                print("An error occurred while training")
                print(e.message)
                self.fitness = -1
                print("Network thrown out")
        self.trained = True

    def mutate(self):
        '''
        Mutates some aspect of the network at random. Must be built afterwards.
        '''
        type = random.randrange(15)
        if type == 0:
            self.period = random.choice(['1m', '30m', '1h', '2h', '6h', '12h'])
        elif type == 1:
            self.look_back = random.randrange(15, 60)
        elif type == 2:
            self.look_ahead = random.randrange(1, 15)
        elif type == 3:
            self.loss = random.choice(["mean_squared_error", "mean_absolute_error", "mean_squared_logarithmic_error",
                                   "squared_hinge", "sparse_categorical_crossentropy", "kullback_leibler_divergence",
                                   "cosine_proximity"])
        elif type == 4:
            self.optimizer = random.choice(
                [optimizers.Nadam(), optimizers.SGD(), optimizers.RMSprop(), optimizers.Adagrad()])
        elif type == 5:
            if len(self.layers) >= 2:
                self.layers.pop(random.randrange(len(self.layers)))
            else:
                self.layers.insert(random.randrange(0, len(self.layers)), self.randomLayer())
        elif type == 6:
            self.layers.insert(random.randrange(0, len(self.layers)), self.randomLayer())
        elif type == 7:
            self.layers[random.randrange(0, len(self.layers))] = self.randomLayer()
        elif type == 8:
            self.layers[0] = self.randomLayer(top=True)
        elif type == 9:
            self.symbols = self.randomSymbolSet()
        elif type == 10:
            self.train_epochs = random.randrange(50, 200)
        else:
            l = random.randrange(0, len(self.layers))
            self.layers[l] = (self.layers[l][0], self.layers[l][1], self.layers[l][2] + random.randrange(-257, 257), self.layers[l][3], self.layers[l][4])

    def hypermutate(self, repetitions = 3):
        '''
        Mutates multiple times
        '''
        for i in range(repetitions):
            self.mutate()

    def randomLayer(self, top=False):
        '''
        Creates a randomly generated layer
        '''
        layer = [None, None, None, None, None]
        if top:
            layer[0] = random.choice(["Dense", "Conv", "Conv+Pool", "LSTM", "Reshape"])
        else:
            layer[0] = random.choice(
                ["Dense", "Conv", "Conv+Pool", "LSTM", "GausNoise", "Dropout", "BatchNorm", "Reshape", "Flatten"])

        layer[1] = random.choice(["sigmoid", "relu", "tanh", "softmax"])
        layer[2] = random.randrange(12, 257)
        layer[3] = random.randrange(99)/100 + 1

        layer[4] = bool(random.getrandbits(1))
        return tuple(layer)

    def randomSymbolSet(self, top=False):
        '''
        Randomly selects a set of exchange symbols
        '''
        symbols = []
        for l in range(random.randrange(1, 6)):
            while 1:
                rando = random.choice(['BTC/USDT', 'ETH/USDT', 'LINK/BTC', 'XLM/BTC', 'TRX/BTC', 'XMR/BTC', 'ZEC/BTC', 'LTC/BTC', 'BNB/BTC'])
                if rando not in symbols:
                    symbols.append(rando)
                    break
                else:
                    continue
        return symbols

    def print_plot(self):
        if self.train_plot != None:
            self.train_plot.savefig(self.name + '.png', bbox_inches='tight')


class Population():
    '''
        Represents a population of networks which can breed together and evolve
    '''
    def __init__(self, count, testmode = False, population_file=None):
        self.testmode = testmode
        self.pop = []
        self.generation = 0
        self.net_retain_pct = .4
        self.reject_survival_pct = .1
        # (Out of 10)
        self.mutate_chance = 6
        self.hypermutate_chance = 9
        while len(self.pop) < count:
            n = Network()
            n.name = random_name()
            n.randomize()

            trainable_layers = 0
            for layer in n.layers:
                if layer[0] == "Dense" or layer[0] == "Conv" or layer[0] == "Conv+Pool" or layer[0] == "LSTM":
                    trainable_layers += 1
            if trainable_layers == 0:
                del n.network
                del n
                gc.collect()
                continue

            if testmode:
                n.train_epochs = 2
                n.layers = [('Dense', 'sigmoid', 62, 1, True), ('Flatten', 'sigmoid', 86, 1, True)]

            n.train(quicktrain=True)
            if n.fitness != -1 or testmode:
                self.pop.append(n)
            else:
                del n.network
                del n
                gc.collect()

    def save_pop(self):
        pop = tuple(self.pop)
        for net in pop:
            net.trained = False
            net.network = None
            net.train_plot = None
        file = open("pop.obj", "wb")
        pickle.dump(pop, file, pickle.HIGHEST_PROTOCOL)

    def load_pop(self):
        file = open("pop.obj", "wb")
        pop = pickle.load(file)
        for net in pop:
            net.trained = False
            net.network = Sequential()
            net.train()
        self.pop = list(pop)

    def breed(self, mother, father):
        '''
        Take two nets and create combined (and sometimes mutated) "children" which are combinations of them.
        '''
        children = []
        for i in range(random.randrange(1, 5)):

            child = Network()

            child.name = random_name()
            child.period =  random.choice([mother.period, father.period])
            child.look_back = random.choice([mother.look_back, father.look_back])
            child.look_ahead = random.choice([mother.look_ahead, father.look_ahead])
            child.symbols = random.choice([mother.symbols, father.symbols])
            child.layers = random.choice([mother.layers, father.layers])
            child.loss = random.choice([mother.loss, father.loss])
            child.optimizer = random.choice([mother.optimizer, father.optimizer])

            #Randomly mutate some children
            if random.randrange(10) > self.mutate_chance:
                child.mutate()
            elif random.randrange(10) > self.hypermutate_chance:
                child.hypermutate()

            children.append(child)

        return children

    def evolve(self):
        """
        Evolve the population.


        Note - Currently populations are creating children that cannot be trained and then allowing them to replace
        their working parents until no working networks are left and all populations are shot through rapidly because they
        cannot be trained. Some kind of measure should be put in place to ensure the parents will be reused if none
        of the children are viable.
        """
        for network in self.pop:
            if not network.trained:
             network.train()


        #Get grades and overall grade
        graded = [(network.fitness, network) for network in self.pop]
        grade = numpy.average([n[0] for n in graded])

        #Make list of networks sorted by grade
        graded = [x[1] for x in sorted(graded, key=lambda x: x[0], reverse=True)]

        #Outputs
        print("Generation " + str(self.generation) + "Population grade: " + str(grade) + ", Top Net grade: " + str(graded[0].fitness))
        graded[0].print_plot()

        #Keep the best ones as parents
        retain = int(len(graded)*self.net_retain_pct)
        parents = graded[:retain]

        for net in graded[retain:]:
            if self.reject_survival_pct > random.random():
                parents.append(net)

        # Now find out how many spots we have left to fill.
        children_needed = len(self.pop) - len(parents)
        children = []

        # Add children, which are bred from two remaining networks.
        while len(children) < children_needed:

            # Get a random mom and dad.
            mom = random.choice(parents)
            dad = random.choice(parents)

            #Breed and keep the children only if they can be built
            if dad != mom:
                babies = self.breed(mom, dad)
                for baby in babies:
                    if len(children) < children_needed:
                        if self.testmode:
                            baby.train_epochs = 2
                            baby.layers = [('Dense', 'sigmoid', 62, 1, True), ('Flatten', 'sigmoid', 86, 1, True)]
                        baby.train()
                        if baby.fitness == -1:
                            continue
                        children.append(baby)
                    else:
                        break

        parents.extend(children)

        self.generation += 1
        self.pop = parents


if __name__ == "__main__":
    p = Population(20)
    #p.save_pop()
    for i in range(30):
        p.evolve()