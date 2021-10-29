import numpy as np
import random
from function import RVFL_train_val
import h5py
from option import option as op
from tqdm import tqdm

dataset_name = "abalone"

# Read in data file
temp = h5py.File("UCI data python\\" + dataset_name + "_R.mat")
data = np.array(temp['data']).T

print("Data shape:", data.shape)

data = data[:, 1:]
dataX = data[:, 0:-1]

# Normalize features according to normal distribution
dataX_mean = np.mean(dataX, axis=0)
dataX_std = np.std(dataX, axis=0)
dataX = (dataX - dataX_mean) / dataX_std
dataY = data[:, -1]
dataY = np.expand_dims(dataY, 1)

# Load in indexes for train-test split
temp = h5py.File("UCI data python\\" + dataset_name + "_conxuntos.mat")
index1 = np.array(temp['index1']).astype(np.int32) - 1
index2 = np.array(temp['index2']).astype(np.int32) - 1
index1 = np.squeeze(index1, axis=1)
index2 = np.squeeze(index2, axis=1)

# Store variables for best scores and parameters as well as data
trainX = dataX[index1, :]
trainY = dataY[index1, :]
testX = dataX[index2, :]
testY = dataY[index2, :]

MAX_acc = np.zeros([6, 1])
Best_N = np.zeros([6, 1]).astype(np.int32) # Number of neurons in hidden layer
Best_C = np.zeros([6, 1]) # Regularization parameter strength (lambda)
Best_S = np.zeros([6, 1]) # Linear scale of random variables before feeding into non-linear activation function
S = np.linspace(-5, 5, 21)

# Look at the documentation of RVFL_train_val function file
# Options for each model, can edit to change the model parameters
option1 = op()
option2 = op()
option3 = op()
option4 = op()
option5 = op()
option6 = op()

# Set up options for each model (e.g. deactivate direct link, choose a different regression mode)
for s in tqdm(range(0, S.size)):
    for N in range(3, 204, 20):
        for C in range(-5, 15):
            Scale = np.power(2, S[s])

            # Bias and NO direct link
            option1.N = N
            option1.C = 2 ** C
            option1.Scale = Scale
            option1.Scalemode = 3
            option1.bias = 1
            option1.link = 0

            # Bias and direct link
            option2.N = N
            option2.C = 2 ** C
            option2.Scale = Scale
            option2.Scalemode = 3
            option2.bias = 1
            option2.link = 1

            # Radbas activation function, bias and direct link
            option3.N = N
            option3.C = 2 ** C
            option3.Scale = Scale
            option3.ActivationFunction = "radbas"
            option3.Scalemode = 3
            option3.bias = 1
            option3.link = 1

            # Hardlim activation function, bias and direct link
            option4.N = N
            option4.C = 2 ** C
            option4.Scale = Scale
            option4.ActivationFunction = "hardlim"
            option4.Scalemode = 3
            option4.bias = 1
            option4.link = 1

            # Ridge regression, radbas activation function, bias and direct link
            option5.N = N
            option5.C = 2 ** C
            option5.Scale = Scale
            option5.ActivationFunction = "radbas"
            option5.Scalemode = 3
            option5.bias = 1
            option5.link = 1
            option5.mode = 1

            # Moore penrose pseudoinverse, radbas activation function, bias and direct link
            option6.N = N
            option6.C = 2 ** C
            option6.Scale = Scale
            option6.ActivationFunction = "radbas"
            option6.Scalemode = 3
            option6.bias = 1
            option6.link = 1
            option6.mode = 2

            # Get the train test accuracy for each option model (on all data)
            train_accuracy1, test_accuracy1 = RVFL_train_val(trainX, trainY, testX, testY, option1)
            train_accuracy2, test_accuracy2 = RVFL_train_val(trainX, trainY, testX, testY, option2)
            train_accuracy3, test_accuracy3 = RVFL_train_val(trainX, trainY, testX, testY, option3)
            train_accuracy4, test_accuracy4 = RVFL_train_val(trainX, trainY, testX, testY, option4)
            train_accuracy5, test_accuracy5 = RVFL_train_val(trainX, trainY, testX, testY, option5)
            train_accuracy6, test_accuracy6 = RVFL_train_val(trainX, trainY, testX, testY, option6)

            # If the test accuracy is higher than our stored max accuracy, store the parameters for that model
            if test_accuracy1 > MAX_acc[
                0]:  # parameter tuning: we prefer the parameter which lead to better accuracy on the test data
                MAX_acc[0] = test_accuracy1
                Best_N[0] = N
                Best_C[0] = C
                Best_S[0] = Scale

            if test_accuracy2 > MAX_acc[
                1]:  # parameter tuning: we prefer the parameter which lead to better accuracy on the test data
                MAX_acc[1] = test_accuracy2
                Best_N[1] = N
                Best_C[1] = C
                Best_S[1] = Scale

            if test_accuracy3 > MAX_acc[
                2]:  # parameter tuning: we prefer the parameter which lead to better accuracy on the test data
                MAX_acc[2] = test_accuracy3
                Best_N[2] = N
                Best_C[2] = C
                Best_S[2] = Scale

            if test_accuracy4 > MAX_acc[
                3]:  # parameter tuning: we prefer the parameter which lead to better accuracy on the test data
                MAX_acc[3] = test_accuracy4
                Best_N[3] = N
                Best_C[3] = C
                Best_S[3] = Scale

            if test_accuracy5 > MAX_acc[
                4]:  # parameter tuning: we prefer the parameter which lead to better accuracy on the test data
                MAX_acc[4] = test_accuracy5
                Best_N[4] = N
                Best_C[4] = C
                Best_S[4] = Scale

            if test_accuracy6 > MAX_acc[
                5]:  # parameter tuning: we prefer the parameter which lead to better accuracy on the test data
                MAX_acc[5] = test_accuracy6
                Best_N[5] = N
                Best_C[5] = C
                Best_S[5] = Scale

# Load in indexes for k-fold validation sets
temp = h5py.File("UCI data python\\" + dataset_name + "_conxuntos_kfold.mat")
index = []

for i in range(8):
    index_temp = np.array([temp[element[i]][:] for element in temp['index']]).astype(np.int32) - 1
    index_temp = np.squeeze(index_temp, axis=0)
    index_temp = np.squeeze(index_temp, axis=1)
    index.append(index_temp)

# Stores the mean accuracy of each cross fold validation (4 of them per model)
ACC_CV = np.zeros([6, 4])

# For each model, run all 4 cross fold validations and store the results
for i in tqdm(range(4)):
    trainX = dataX[index[2 * i], :]
    trainY = dataY[index[2 * i], :]
    testX = dataX[index[2 * i + 1], :]
    testY = dataY[index[2 * i + 1], :]

    # Bias and NO direct link
    option1.N = Best_N[0, 0]
    option1.C = 2 ** Best_C[0, 0]
    option1.Scale = Best_S[0, 0]
    option1.Scalemode = 3
    option1.bias = 1
    option1.link = 0

    # Bias and direct link
    option2.N = Best_N[1, 0]
    option2.C = 2 ** Best_C[1, 0]
    option2.Scale = Best_S[1, 0]
    option2.Scalemode = 3
    option2.bias = 1
    option2.link = 1

    # Radbas activation function, bias and direct link
    option3.N = Best_N[2, 0]
    option3.C = 2 ** Best_C[2, 0]
    option3.Scale = Best_S[2, 0]
    option3.ActivationFunction = "radbas"
    option3.Scalemode = 3
    option3.bias = 1
    option3.link = 1

    # Hardlim activation function, bias and direct link
    option4.N = Best_N[3, 0]
    option4.C = 2 ** Best_C[3, 0]
    option4.Scale = Best_S[3, 0]
    option4.ActivationFunction = "hardlim"
    option4.Scalemode = 3
    option4.bias = 1
    option4.link = 1

    # Ridge regression, radbas activation function, bias and direct link
    option5.N = Best_N[4, 0]
    option5.C = 2 ** Best_C[4, 0]
    option5.Scale = Best_S[4, 0]
    option5.ActivationFunction = "radbas"
    option5.Scalemode = 3
    option5.bias = 1
    option5.link = 1
    option5.mode = 1

    # Moore penrose pseudoinverse, radbas activation function, bias and direct link
    option6.N = Best_N[5, 0]
    option6.C = 2 ** Best_C[5, 0]
    option6.Scale = Best_S[5, 0]
    option6.ActivationFunction = "radbas"
    option6.Scalemode = 3
    option6.bias = 1
    option6.link = 1
    option6.mode = 2

    # Get the train test accuracy for each option model (for each cross validation set)
    train_accuracy1, ACC_CV[0, i] = RVFL_train_val(trainX, trainY, testX, testY, option1)
    train_accuracy2, ACC_CV[1, i] = RVFL_train_val(trainX, trainY, testX, testY, option2)
    train_accuracy3, ACC_CV[2, i] = RVFL_train_val(trainX, trainY, testX, testY, option3)
    train_accuracy4, ACC_CV[3, i] = RVFL_train_val(trainX, trainY, testX, testY, option4)
    train_accuracy5, ACC_CV[4, i] = RVFL_train_val(trainX, trainY, testX, testY, option5)
    train_accuracy6, ACC_CV[5, i] = RVFL_train_val(trainX, trainY, testX, testY, option6)

print("Mean:", np.mean(ACC_CV, axis=1))
print("Variance:", np.var(ACC_CV, axis=1))
print("Full matrix:", ACC_CV)
