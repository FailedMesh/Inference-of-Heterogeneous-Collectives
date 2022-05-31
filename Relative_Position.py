import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from IPython.display import clear_output
import scipy.io
import os
from Helper_Functions import *
import time
import random


class relative_position_model:

    def __init__(self, fixed_params, unique_params, files_path, folder_name):
        self.fixed_params = fixed_params
        file_names = os.listdir(files_path)
        file_params = list(map(get_sim_data, file_names))
        self.file_names = parametric_sim_data(file_names, file_params, unique_params, fixed_params)
        self.folder_name = folder_name
        self.X = None
        self.Y = None
        self.val_data = None
        self.network = None
        self.epochs = 0
        self.loss = 0

    def initialize(self, 
                    train_ratio, 
                    num_instances = "all"):
    
        file_names = np.array(self.file_names)
        folder_name = self.folder_name
        train_barrier = round(train_ratio*len(file_names))
        folder = file_names[np.random.permutation(len(file_names))][:train_barrier]
        val_data = file_names[train_barrier:]

        #Specify number of files to process:
        file_num = len(folder)
        print("Files to process = ", file_num)
        _ = input("Press Enter to continue")

        file = scipy.io.loadmat(folder_name + folder[0])

        #Initialize the array for the XA and YA positions
        all_Xpos = np.array(file['XA']).T
        all_Ypos = np.array(file['YA']).T
        if num_instances != "all":
            perm = np.random.permutation(len(all_Xpos))
            all_Xpos = all_Xpos[perm[:num_instances]]
            all_Ypos = all_Ypos[perm[:num_instances]]

        #Get the Velocities along X-direction for current file:
        VXA = np.array(file['VXA'])
        
        #Extract the first timestep, the initial velocity, and divide it by its magnitude to get the direction (1 or -1)
        ivel = (VXA[:, 0]/abs(VXA[0, 0]))
        ivel = ((ivel + 1)/2).astype('int')

        X, Y = self.transform_to_relative(all_Xpos, all_Ypos, ivel)

        processed = 1

        for i in range(1, file_num):

            file = scipy.io.loadmat(folder_name + folder[i])

            #Initialize the array for the XA and YA positions
            all_Xpos = np.array(file['XA']).T
            all_Ypos = np.array(file['YA']).T
            if num_instances != "all":
                perm = np.random.permutation(len(all_Xpos))
                all_Xpos = all_Xpos[perm[:num_instances]]
                all_Ypos = all_Ypos[perm[:num_instances]]

            #Get the Velocities along X-direction for current file:
            VXA = np.array(file['VXA'])

            #Extract the first timestep, the initial velocity, and divide it by its magnitude to get the direction (1 or -1)
            ivel = (VXA[:, 0]/abs(VXA[0, 0]))
            ivel = ((ivel + 1)/2).astype('int')

            X_curr, Y_curr = self.transform_to_relative(all_Xpos, all_Ypos, ivel)

            X = np.concatenate((X, X_curr), axis = 0)
            Y = np.concatenate((Y, Y_curr), axis = 0)

            processed += 1
            print("Files read = ", processed)
            clear_output(wait=True)
        
        self.X = X[:]
        self.Y = Y[:]
        self.val_data = val_data[:]

    def neural_network(self,
                layer_nodes, 
                dropout_rate,
                lr = 0.001,
                loss = 'binary_crossentropy'):

        inputs = tf.keras.layers.Input(shape = (82,))
        layer_output = tf.keras.layers.Dense(units = layer_nodes[0], activation = 'tanh')(inputs)
        layer_output = tf.keras.layers.Dropout(0)(layer_output)
        layer_output = tf.keras.layers.BatchNormalization()(layer_output)

        for i in range(1, len(layer_nodes)-1):
            layer_output = tf.keras.layers.Dense(units = layer_nodes[i], activation = 'tanh')(layer_output)
            layer_output = tf.keras.layers.Dropout(dropout_rate)(layer_output)
            layer_output = tf.keras.layers.BatchNormalization()(layer_output)
            
        layer_output = tf.keras.layers.Dense(units = layer_nodes[-1], activation = 'sigmoid')(layer_output)

        self.network = tf.keras.Model(inputs = inputs, outputs = layer_output)

        self.network.compile(optimizer = tf.keras.optimizers.Adam(learning_rate = lr), 
              loss = loss,
             metrics = [tf.keras.metrics.Accuracy()])

    def transform_to_relative(self, X_pos, Y_pos, ivel):
        all_rel_X = np.array([])
        all_rel_Y = np.array([])
        labels = np.array([])
        for index in range(X_pos.shape[1]):
            rel_X, rel_Y = get_relative_positions(X_pos, Y_pos, index)
            labels = np.append(labels, np.repeat(ivel[index], X_pos.shape[0]))
            all_rel_X = np.reshape(np.append(all_rel_X, rel_X), [-1, rel_X.shape[1]])
            all_rel_Y = np.reshape(np.append(all_rel_Y, rel_Y), [-1, rel_Y.shape[1]])
        all_rel_pos = np.concatenate([all_rel_X, all_rel_Y], axis = 1)
        labels = np.reshape(labels, [all_rel_pos.shape[0], 1])
        return all_rel_pos, labels

    def load_model(self, model_name):
        self.network = tf.keras.models.load_model(model_name, compile = False)
        self.val_data = self.file_names[:]

    def train(self, batch_size = "all", epochs = 2000):

        if batch_size == "all":
            batch_size = self.X.shape[0]
        self.history = self.network.fit(x = self.X, y = self.Y, batch_size = batch_size, epochs = epochs)
        self.epochs += epochs
        self.loss = self.history.history['loss'][-1]
        print("MODEL STATUS")
        print("Model has been trained for a total of ", self.epochs, " epochs")
        print("Model Loss = ", self.loss)
        plt.plot(self.history.history['loss'])
        plt.title('loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'val'], loc='upper left')
        plt.show()

    def generate_prediction(self, eval_file = "random"):

        if eval_file == "random":    
            file_name = self.folder_name + random.choice(self.val_data)
        else:
            file_name = self.folder_name + eval_file 
        file = scipy.io.loadmat(file_name)
        X_position = np.array(file['XA']).T
        Y_position = np.array(file['YA']).T
        VXA = file['VXA']
        Y = (VXA[:, 0]/abs(VXA[0, 0]))
        Y = np.reshape(((Y + 1)/2).astype('int'), [1, -1])
        Y = np.tile(Y, (VXA.shape[1], 1))
        rel_X, rel_Y = get_relative_positions(X_position, Y_position, 0)
        Input = np.concatenate([rel_X, rel_Y], axis = 1)
        self.Output = self.network.predict(Input)
        for particle in range(1, X_position.shape[1]):
            rel_X, rel_Y = get_relative_positions(X_position, Y_position, particle)
            Input = np.concatenate([rel_X, rel_Y], axis = 1)
            self.Output = np.concatenate([self.Output, self.network.predict(Input)], axis = 1)
        self.rounded_output = np.round(self.Output)
        self.error = abs(Y - self.rounded_output)
        num_misclassifications = np.sum(self.error)
        print("NUMBER OF MISCLASSIFICATIONS = ", num_misclassifications)
        raw_error = np.sum(abs(Y - self.Output))/X_position.size
        self.accuracy = (1 - raw_error)*100
        print("ACCURACY = ", self.accuracy)
        print("DATASET USED = " + file_name)

        return self.rounded_output, self.error




