import numpy as np
import random
from function import RVFL_train_val
import h5py
from option import option as op

dataset_name = "adult"

temp1 = h5py.File("UCI data python\\" + dataset_name + "_train_R.mat")
data1 = np.array(temp1['data1']).T

temp2 = h5py.File("UCI data python\\" + dataset_name + "_test_R.mat")
data2 = np.array(temp2['data2']).T

# do normalization for each feature
data = np.concatenate((data1, data2), axis=0)
N = data1.shape[0]

data = data[:, 1:]
dataX = data[:, 0:-1]
dataX_mean = np.mean(dataX, axis=0)
dataX_std = np.std(dataX, axis=0)
dataX = (dataX - dataX_mean) / dataX_std
dataY = data[:, -1]
dataY = np.expand_dims(dataY, 1)

trainX = dataX[0:N, :]
trainY = dataY[0:N, :]
testX = dataX[N:, :]
testY = dataY[N:, :]

# for some data where the training-testing partitioning is available, the  parameter is tuned on the training set and evaluated on the test data.
temp = h5py.File(
    "UCI data python\\" + dataset_name + "_conxuntos.mat")  # divide the training data into two parts to do paramater tuning.
index1 = np.array(temp['index1']).astype(np.int32) - 1
index2 = np.array(temp['index2']).astype(np.int32) - 1
index1 = np.squeeze(index1, axis=1)
index2 = np.squeeze(index2, axis=1)

trainX_temp = dataX[index1, :]
trainY_temp = dataY[index1, :]
testX_temp = dataX[index2, :]
testY_temp = dataY[index2, :]

MAX_acc = np.zeros([4, 1])
Best_N = np.zeros([4, 1]).astype(np.int32)
Best_C = np.zeros([4, 1])
Best_S = np.zeros([4, 1])
S = np.linspace(-5, 5, 21)
# # Look at the documentation of RVFL_train_val function file
option1 = op()
option2 = op()
option3 = op()
option4 = op()

for s in range(0, S.size):

    for N in range(3, 204, 20):

        for C in range(-5, 15):

            Scale = np.power(2, S[s])
            option1.N = N
            option1.C = 2 ** C
            option1.Scale = Scale
            option1.Scalemode = 3
            option1.bias = 0
            option1.link = 0

            option2.N = N
            option2.C = 2 ** C
            option2.Scale = Scale
            option2.Scalemode = 3
            option2.bias = 1
            option2.link = 0

            option3.N = N
            option3.C = 2 ** C
            option3.Scale = Scale
            option3.Scalemode = 3
            option3.bias = 0
            option3.link = 1

            option4.N = N
            option4.C = 2 ** C
            option4.Scale = Scale
            option4.Scalemode = 3
            option4.bias = 1
            option4.link = 1

            train_accuracy1, test_accuracy1 = RVFL_train_val(trainX_temp, trainY_temp, testX_temp, testY_temp, option1)
            train_accuracy2, test_accuracy2 = RVFL_train_val(trainX_temp, trainY_temp, testX_temp, testY_temp, option2)
            train_accuracy3, test_accuracy3 = RVFL_train_val(trainX_temp, trainY_temp, testX_temp, testY_temp, option3)
            train_accuracy4, test_accuracy4 = RVFL_train_val(trainX_temp, trainY_temp, testX_temp, testY_temp, option4)

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


ACC_CV = np.zeros([4, 4])

for i in range(4):

    option1.N = Best_N[0, 0]
    option1.C = 2 ** Best_C[0, 0]
    option1.Scale = Best_S[0, 0]
    option1.Scalemode = 3
    option1.bias = 0
    option1.link = 0

    option2.N = Best_N[1, 0]
    option2.C = 2 ** Best_C[1, 0]
    option2.Scale = Best_S[1, 0]
    option2.Scalemode = 3
    option2.bias = 1
    option2.link = 0

    option3.N = Best_N[2, 0]
    option3.C = 2 ** Best_C[2, 0]
    option3.Scale = Best_S[2, 0]
    option3.Scalemode = 3
    option3.bias = 0
    option3.link = 1

    option4.N = Best_N[3, 0]
    option4.C = 2 ** Best_C[3, 0]
    option4.Scale = Best_S[3, 0]
    option4.Scalemode = 3
    option4.bias = 1
    option4.link = 1

    train_accuracy1, ACC_CV[0, i] = RVFL_train_val(trainX, trainY, testX, testY, option1)
    train_accuracy2, ACC_CV[1, i] = RVFL_train_val(trainX, trainY, testX, testY, option2)
    train_accuracy3, ACC_CV[2, i] = RVFL_train_val(trainX, trainY, testX, testY, option3)
    train_accuracy4, ACC_CV[3, i] = RVFL_train_val(trainX, trainY, testX, testY, option4)

print(np.mean(ACC_CV, axis=0))
