#############################################

# Example code on real Right heart catheterization dataset

#############################################

import numpy as np
import pandas as pd
from sklearn import preprocessing
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression



# read data
df = pd.read_csv(
    '../data/Right_heart_catheterization_dataset/rhc.csv',
    sep=','
)

# data wrangling
newdf = df.filter(['death', 'age', 'sex', 'edu', 'surv2md1', 'das2d3pc', 'aps1',
                   'scoma1', 'meanbp1', 'wblc1', 'hrt1', 'resp1', 'temp1',
                   'pafi1', 'alb1', 'hema1', 'bili1', 'crea1', 'sod1', 'pot1',
                   'paco21', 'ph1', 'wtkilo1', 'race', 'income'], axis=1)

newdf['death'] = newdf['death'].map({'No': 0, 'Yes': 1})
newdf['sex'] = newdf['sex'].map({'Male': 0, 'Female': 1})
newdf['race'] = newdf['race'].map({'white': 0, 'other': 1, 'black': 2})
newdf['income'] = newdf['income'].map({'Under $11k': 0, '$25-$50k': 1, '$11-$25k': 2, '> $50k': 3})

# got a dataframe with 5735 examples and 25 variables
newdf.dropna(inplace=True)

# rescale dataframe
min_max_scaler = preprocessing.MinMaxScaler()
newdf[['age', 'edu', 'surv2md1', 'das2d3pc', 'aps1', 'scoma1', 'meanbp1', 'wblc1', 'hrt1', 'resp1', 'temp1',
       'pafi1', 'alb1', 'hema1', 'bili1', 'crea1', 'sod1', 'pot1', 'paco21', 'ph1', 'wtkilo1', 'race', 'income']] \
    = min_max_scaler.fit_transform(newdf[['age', 'edu', 'surv2md1', 'das2d3pc', 'aps1', 'scoma1', 'meanbp1', 'wblc1',
                                          'hrt1', 'resp1', 'temp1', 'pafi1', 'alb1', 'hema1', 'bili1', 'crea1', 'sod1',
                                          'pot1', 'paco21', 'ph1', 'wtkilo1', 'race', 'income']])

### split dataframe into training and test dataset

X = newdf.drop(['death'], axis=1).to_numpy()
y = newdf['death'].to_numpy()

## set parameters
tuning_parameter = 0.3
epoch_number = 5
learning_rate = 0.01
n = 50
p = 0.5
np.random.seed(2002)
torch.manual_seed(2002)

# split train and test set
mask = np.random.rand(len(X)) < 0.9
X_train = X[mask]
y_train = y[mask]
X_test = X[~mask]
y_test = y[~mask]

# change train data for neural network
train_loader_x = torch.tensor(X_train).float()
train_loader_y = torch.tensor(y_train).float()
test_loader_x = torch.tensor(X_test).float()
test_loader_y = torch.tensor(y_test).float()

train_loader = DataLoader(TensorDataset(train_loader_x, train_loader_y))
test_loader = DataLoader(TensorDataset(test_loader_x, test_loader_y))


##########################################
## fit LR models to the data
##########################################

# fit logistic regression to the simulated data
lr = LogisticRegression(penalty="none")
lr.fit(X_train, y_train)

## get regression coefficients
w0 = lr.intercept_
w = lr.coef_[0]

## LR prediction for training set
z = np.sum(w * X_test, axis = 1) + w0
h = 1/(1+np.exp(-z))
lr_test = h
lr_round = np.round(lr_test)

print('The accuracy of logistic regression:', round(sum(lr_round == y_test)/len(y_test)*100,3),'%')

LR_accuracy = round(sum(lr_round == y_test)/len(y_test)*100,3)

###############
#random forest
###############

from sklearn.ensemble import RandomForestRegressor

regr = RandomForestRegressor()
regr.fit(X_train, y_train)
y_rf = regr.predict(X_test)
y_rfround = np.round(y_rf)

print('The accuracy of random forest:', round(sum(y_rfround == y_test)/len(y_test)*100,3),'%')

RF_accuracy = round(sum(y_rfround == y_test)/len(y_test)*100,3)

#############
# kernel ridge regression
#############

from sklearn.kernel_ridge import KernelRidge
from sklearn.svm import SVR

clf1 = SVR(kernel='linear', C=0.001)
clf1.fit(X_train, y_train)
y_krl = clf1.predict(X_test)
y_krlround = np.round(y_krl)

print('The accuracy of linear kernel model:', round(sum(y_krlround == y_test)/len(y_test)*100,3),'%')

LK_accuracy = round(sum(y_krlround == y_test)/len(y_test)*100,3)


clf2 = SVR(kernel='poly', C=0.001)
clf2.fit(X_train, y_train)
y_krp = clf2.predict(X_test)
y_krpround = np.round(y_krp)

print('The accuracy of polynomial kernel model:', round(sum(y_krpround == y_test)/len(y_test)*100,3),'%')

PK_accuracy = round(sum(y_krpround == y_test)/len(y_test)*100,3)

LR.append(LR_accuracy)
RF.append(RF_accuracy)
LK.append(LK_accuracy)
PK.append(PK_accuracy)

###############################

# regularised neural networks

###############################

# set seed
# torch.manual_seed(20)

# Logistic regression
# use multiple layers NN
class NN(nn.Module):

    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(24, 50)
        self.linear2 = nn.Linear(50, 100)
        self.linear3 = nn.Linear(100, 30)
        self.linear4 = nn.Linear(30, 10)
        self.linear5 = nn.Linear(10, 5)
        self.sigmoid = nn.Sigmoid()
        self.linear6 = nn.Linear(5, 1)

    def forward(self, x):
        x = self.linear1(x)
        x = F.relu(self.linear2(x))
        x = F.relu(self.linear3(x))
        x = F.relu(self.linear4(x))
        x = F.relu(self.linear5(x))
        x = self.sigmoid(self.linear6(x))
        return x

modelRS = NN()
modelNR = NN()
modelL2 = NN()

# General training step function
def training_step_regularisation(epoch_number, train_loader, model, learning_rate, regularised_factor, regularisation_function, mean, variance, w0, w, n, wd):
    # optimizer
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=wd)
    j = 0
    for epoch in range(epoch_number):  # loop over the dataset multiple times
        running_loss = 0.0
        i = 0
        for data in train_loader:
            inputs, labels = data
            # reshape train data y
            inputs = inputs.view(-1)
            labels = labels.view(-1)

            # zero the parameter gradients
            optimizer.zero_grad()
            # Forward pass
            outputs = model(inputs)

            # # predicted output
            h1, predicted = regularisation_function(mean, variance, w0, w, model, n)

            # Compute Loss with comparison
            loss = criterion(outputs, labels) + tuning_parameter * regularised_factor * criterion(predicted, h1)
            # Backward pass
            loss.backward()
            # optimize
            optimizer.step()


            # print statistics
            running_loss += loss.item()
            if i % len(train_loader) == (len(train_loader)-1):
                print('[%d, %5d] loss: %.3f' %
                    (epoch + 1, i + 1, running_loss / n))
                a = running_loss/len(train_loader)
            i += 1
        j += 1

    return(model, a)

# acceptance factor function
def random_u(p):
    u = np.random.uniform(0, 1)
    if all(i < u for i in p):
        return True
    else:
        return False

# rejection sampling algorithm of 200 points

def regularisation_RS(mean, variance, w0, w, model, n):
    # initial set true value
    xtrue = torch.empty(0, 24)

    # sample 200 random points
    x = 8 * (torch.rand(n, 24) - 0.5)

    # set numerator and denominator
    numerator = torch.exp((-(x - mean) ** 2) / (2 * variance))
    torch.pi = torch.acos(torch.zeros(1)) * 2
    denominator = torch.sqrt(2 * torch.pi * variance)

    density = numerator / denominator

    maximum_density = 1/denominator

    density = density/maximum_density

    for i in range(n):
        # set proposal distribution
        current = x[i,:].view(-1,24)
        #
        current_numpy = current.numpy()[-1]
        # restrict boundary
        if any(abs(i) > 1.96 for i in current_numpy):
            xtrue = torch.cat([xtrue, current], 0)
            continue

        if random_u(density[i,:]):
            xtrue = torch.cat([xtrue, current], 0)

    z1 = torch.sum(torch.tensor(w) * xtrue, dim=1) + torch.tensor(w0)
    h1 = 1 / (1 + torch.exp(-z1))
    h1 = h1.view(-1,1)

    # prediction
    predicted = model(xtrue)
    predicted = predicted.double()

    return (h1, predicted)

# rejection resampling
modelRS, loss_RS = training_step_regularisation(epoch_number, train_loader, modelRS, learning_rate, 1, regularisation_RS, 0, 1, w0, w, n, wd=0)
modelNR, loss_NR = training_step_regularisation(epoch_number, train_loader, modelNR, learning_rate, 0, regularisation_RS, 0, 1, w0, w, n, wd=0)
modelL2, loss_L2 = training_step_regularisation(epoch_number, train_loader, modelL2, learning_rate, 0, regularisation_RS, 0, 1, w0, w, n, wd=0.001)
######################################

# test the accuracy of neural networks

######################################

def test_accuracy(model):
    correct = 0
    total = 0
    with torch.no_grad():
        for data in test_loader:
            inputs, labels = data
            # reshape train data y
            inputs = inputs.view(-1)
            labels = labels.view(-1)

            # Forward pass
            outputs = model(inputs)

            predicted = torch.round(outputs)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print('Accuracy of the neural network:',round(100 * correct / total,2),'%')
    return(round(100 * correct / total,2))
    # print('Accuracy of the neural network: %d %%' % (
    #     100 * correct / total))

print('NN Accuracy')
print('Regularised:')
test_accuracy(modelRS)
print('Not regularised')
test_accuracy(modelNR)
print('L2 regularisation(weight_decay)')
test_accuracy(modelL2)

print('Training loss:', loss_RS)

NN_accuracy = test_accuracy(modelNR)
L2_accuracy = test_accuracy(modelL2)
BCE_accuracy = test_accuracy(modelRS)

NR.append(NN_accuracy)
L2.append(L2_accuracy)
BCE.append(BCE_accuracy)

# NN = []
# L2 = []
# BCE = []



#
# ##################################
#
# # plot the difference
#
# ##################################
#
# # plot decision boundaries
# def plot_boundary_grid(min, max):
#     ## generate a grid of input points
#     xx1 = (np.arange((max-min)*10) - abs(min)*10)/10
#     xx2 = (np.arange((max-min)*10) - abs(min)*10)/10
#     df_xx1 = [[x, y] for x in xx1 for y in xx2]
#     df_xx = np.array(df_xx1)
#
#     return(df_xx)
#
# # decide boundary range
# df_xx = plot_boundary_grid(-4,4)
#
# ## get logistic regression prediction at these grid points
# z_xx = w0 + w[0] * df_xx[:,0] + w[1] * df_xx[:,1]
# h_xx = 1/(1 + np.exp(-z_xx))
# lr_pred = np.round(h_xx)
#
# # get prediction for random forest
# y_rf = regr.predict(df_xx)
# rf_pred = np.round(y_rf)
#
# # get prediction for kernel ridge regression
# y_krl = clf1.predict(df_xx)
# krl_pred = np.round((y_krl-np.min(y_krl))/(np.max(y_krl)-np.min(y_krl)))
# y_krp = clf2.predict(df_xx)
# krp_pred = np.round((y_krp-np.min(y_krp))/(np.max(y_krp)-np.min(y_krp)))
#
# # run nn in 2 dimensional
# # change train data for neural network
# train_loader_xpca = torch.tensor(X_pca).float()
# train_loader_ypca = torch.tensor(y_train).float()
# train_loader = DataLoader(TensorDataset(train_loader_xpca, train_loader_ypca))
#
# # Logistic regression
# # use multiple layers NN
# class NN(nn.Module):
#
#     def __init__(self):
#         super().__init__()
#         self.linear1 = nn.Linear(2, 10)
#         self.linear2 = nn.Linear(10, 20)
#         self.linear3 = nn.Linear(20, 50)
#         self.linear4 = nn.Linear(50, 20)
#         self.linear5 = nn.Linear(20, 10)
#         self.linear6 = nn.Linear(10, 5)
#         self.sigmoid = nn.Sigmoid()
#         self.linear7 = nn.Linear(5, 1)
#
#     def forward(self, x):
#         x = self.linear1(x)
#         x = F.relu(self.linear2(x))
#         x = F.relu(self.linear3(x))
#         x = F.relu(self.linear4(x))
#         x = F.relu(self.linear5(x))
#         x = F.relu(self.linear6(x))
#         x = self.sigmoid(self.linear7(x))
#         return x
#
# modelRS2 = NN()
#
# # General training step function
# def training_step_regularisation2(epoch_number, train_loader, model, learning_rate, regularised_factor, regularisation_function, mean, variance, w0, w, n):
#     # terms for further check
#     average_loss = np.zeros(epoch_number)
#     # optimizer
#     criterion = nn.BCELoss()
#     optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
#     j = 0
#     for epoch in range(epoch_number):  # loop over the dataset multiple times
#         running_loss = 0.0
#         i = 0
#         b = 0
#         for data in train_loader:
#             inputs, labels = data
#             # reshape train data y
#             inputs = inputs.view(-1)
#             labels = labels.view(-1)
#
#             # zero the parameter gradients
#             optimizer.zero_grad()
#             # Forward pass
#             outputs = model(inputs)
#
#             # # predicted output
#             h1, predicted = regularisation_function(mean, variance, w0, w, model)
#
#             # Compute Loss with comparison
#             loss = criterion(outputs, labels) + tuning_parameter * regularised_factor * criterion(predicted, h1)
#             # Backward pass
#             loss.backward()
#             # optimize
#             optimizer.step()
#
#
#             # print statistics
#             running_loss += loss.item()
#             if i % len(train_loader) == (len(train_loader)-1):  # print every 20 mini-batches
#                 print('[%d, %5d] loss: %.3f' %
#                     (epoch + 1, i + 1, running_loss / n))
#                 a = running_loss/len(train_loader)
#             i += 1
#         average_loss[j] = a
#         j += 1
#
#     return(model, average_loss)
#
# # rejection sampling algorithm of 200 points
#
# def regularisation_RS2(mean, variance, w0, w, model):
#     # initial set true value
#     xtrue = torch.empty(0, 2)
#
#     # sample 200 random points
#     x = 8 * (torch.rand(200, 2) - 0.5)
#
#     # set numerator and denominator
#     numerator = torch.exp((-(x - mean) ** 2) / (2 * variance))
#     torch.pi = torch.acos(torch.zeros(1)) * 2
#     denominator = torch.sqrt(2 * torch.pi * variance)
#
#     density = numerator / denominator
#
#     maximum_density = 1/denominator
#
#     density = density/maximum_density
#
#     for i in range(200):
#         # set proposal distribution
#         current = x[i,:].view(-1,2)
#         #
#         current_numpy = current.numpy()[-1]
#         # restrict boundary
#         if any(abs(i) > 1.96 for i in current_numpy):
#             xtrue = torch.cat([xtrue, current], 0)
#             continue
#
#         if random_u(density[i,:]):
#             xtrue = torch.cat([xtrue, current], 0)
#
#     z1 = torch.sum(torch.tensor(w) * xtrue, dim=1) + torch.tensor(w0)
#     h1 = 1 / (1 + torch.exp(-z1))
#     h1 = h1.view(-1,1)
#
#     # prediction
#     predicted = model(xtrue)
#     predicted = predicted.double()
#
#     return (h1, predicted)
#
# # rejection resampling
# modelRS2, average_loss_RS = training_step_regularisation2(epoch_number, train_loader, modelRS2, learning_rate, 1, regularisation_RS2, 0, 1, w0, w, n)
#
#
# # get prediction for neural network
# pred_grid = DataLoader(TensorDataset(torch.tensor(df_xx).float()))
#
# def neural_network_pred_grid(max,min,model,pred_grid):
#     nn_pred_grid = np.zeros(((max-min) ** 2 * 100, 2))
#     i = 0
#     with torch.no_grad():
#         for data in pred_grid:
#             inputs = data
#             outputs = model(inputs[0])
#             nn_pred_grid[i,:] = outputs
#             i = i + 1
#     return(nn_pred_grid)
#
# nn_RS_pred_grid = neural_network_pred_grid(4,-4,modelRS2,pred_grid)
#
# ## store results
# d = {'x1': df_xx[:, 0], 'x2': df_xx[:, 1], 'lr_pred': lr_pred, 'rf_pred': rf_pred, 'krl_pred': krl_pred, 'krp_pred': krp_pred, 'nn_pred': np.round(nn_RS_pred_grid[:,1])}
#
# compare_df = pd.DataFrame(data = d)
#
# ## plot classification grid for LR and different regularised NN
# name = ['Logistic Regression', 'Random forest', 'Kernel-based linear', 'Kernel-based polynomial', 'Neural network']
# plt.subplots(nrows=1, ncols=3)
#
# pred_name = ['lr_pred', 'rf_pred', 'krl_pred', 'krp_pred', 'nn_pred']
# # compare = np.array(compare_df)
#
# for i in range(1,6):
#     plt.subplot(1, 5, i)
#     y0 = training_df.loc[training_df['y'] == 0]
#     y1 = training_df.loc[training_df['y'] == 1]
#
#     pred0 = compare_df.loc[compare_df[pred_name[i - 1]] == 0]
#     pred1 = compare_df.loc[compare_df[pred_name[i - 1]] == 1]
#     c = plt.scatter(np.array(pred0)[:, 0], np.array(pred0)[:, 1], color = 'pink', alpha = 0.1, s = 6)
#     d = plt.scatter(np.array(pred1)[:, 0], np.array(pred1)[:, 1], color = 'c', alpha = 0.1, s = 6)
#
#     a = plt.scatter(np.array(y0)[:, 0], np.array(y0)[:, 1], color='red', alpha = 0.2, s = 3)
#     b = plt.scatter(np.array(y1)[:, 0], np.array(y1)[:, 1], color='g', alpha = 0.2, s = 3)
#
#     plt.title(name[i-1])
#     plt.legend((a, b),
#                ('y = 0', 'y = 1'),
#                scatterpoints=1,
#                loc='upper right',
#                ncol=1,
#                fontsize=8)
#     plt.xlim(-4, 4)
#     plt.ylim(-4, 4)
#     # plt.xlim(-8, 15)
#     # plt.ylim(-8, 15)
#     plt.xlabel('x1', fontsize = 8)
#     plt.ylabel('x2', fontsize = 8)
#
# plt.show()
#
#
# ###############################
# # umap dimension reduction plots
# import matplotlib.pyplot as plt
# import umap
#
# mapperRHC = umap.UMAP(n_neighbors=5,
#                    min_dist=0.3,
#                    metric='correlation').fit(X_test)
#
# def plot_boundary_grid(min, max):
#     ## generate a grid of input points
#     xx1 = (np.arange((max-min)*5) - abs(min)*5)/5
#     xx2 = (np.arange((max-min)*5) - abs(min)*5)/5
#     df_xx1 = [[x, y] for x in xx1 for y in xx2]
#     df_xx = np.array(df_xx1)
#
#     return(df_xx)
# df_xx = plot_boundary_grid(-30,30)
# inv_transformed_points = mapperRHC.inverse_transform(df_xx)
#
# #lr_pred
# z = np.sum(w * inv_transformed_points, axis = 1) + w0
# h = 1/(1+np.exp(-z))
# lr_test = h
# lr_pred_RHC = np.round(lr_test)
#
# # rf pred
# y_rf = regr.predict(inv_transformed_points)
# rf_pred_RHC = np.round(y_rf)
#
# # PKM pred
# y_krp = clf2.predict(inv_transformed_points)
# pkm_pred_RHC = np.round(y_krp)
# pkm_pred_RHC[pkm_pred_RHC < 0] = 0
# pkm_pred_RHC[pkm_pred_RHC > 1] = 1
#
# # neural network pred
# pred_grid = DataLoader(TensorDataset(torch.tensor(inv_transformed_points).float()))
#
# def neural_network_pred_grid(max,min,model,pred_grid):
#     nn_pred_grid = np.zeros(((max-min) ** 2 * 25))
#     i = 0
#     with torch.no_grad():
#         for data in pred_grid:
#             inputs = data
#             outputs = model(inputs[0])
#             nn_pred_grid[i] = outputs
#             i = i + 1
#     return(nn_pred_grid)
#
# NR_pred_grid = neural_network_pred_grid(30,-30,modelNRRHC,pred_grid)
# NR_pred_RHC = np.round(NR_pred_grid)
# RS_pred_grid = neural_network_pred_grid(30,-30,modelRSRHC,pred_grid)
# RS_pred_RHC = np.round(RS_pred_grid)
#
#
# d3 = {'x1': df_xx[:, 0], 'x2': df_xx[:, 1], 'lr_pred': lr_pred_RHC, 'pkm_pred': pkm_pred_RHC, 'NR_pred': NR_pred_RHC, 'RS_pred': RS_pred_RHC}
#
# compare_df3 = pd.DataFrame(data = d3)
# training_df3 = pd.DataFrame(np.c_[mapperRHC.transform(X_test), y_test],columns=['First component', 'Second component', 'y'])
#
# name = ['Logistic Regression', 'Polynomial Kernel', 'NN without Regularisation', 'NN with regularisation']
#
# pred_name = ['lr_pred', 'pkm_pred', 'NR_pred', 'RS_pred']
#
#
# for i in range(1,5):
#     plt.subplot(3, 4, 8+i)
#
#     pred0 = compare_df3.loc[compare_df3[pred_name[i - 1]] == 0]
#     pred1 = compare_df3.loc[compare_df3[pred_name[i - 1]] == 1]
#     c = plt.scatter(np.array(pred0)[:, 0], np.array(pred0)[:, 1], color = 'pink', alpha = 0.1, s = 3)
#     d = plt.scatter(np.array(pred1)[:, 0], np.array(pred1)[:, 1], color = 'c', alpha = 0.1, s = 3)
#     y0 = training_df3.loc[training_df3['y'] == 0]
#     y1 = training_df3.loc[training_df3['y'] == 1]
#     a = plt.scatter(np.array(y0)[:, 0], np.array(y0)[:, 1], color='red',alpha = 0.5, s = 1)
#     b = plt.scatter(np.array(y1)[:, 0], np.array(y1)[:, 1], color='g',alpha = 0.5, s = 1)
#
#     plt.title(name[i-1])
#     plt.legend((a, b),
#                ('y = 0', 'y = 1'),
#                scatterpoints=1,
#                loc='upper right',
#                ncol=1,
#                fontsize=8)
#     plt.xlim(-30, 30)
#     plt.ylim(-30, 30)
#     #plt.xlabel('First component', fontsize = 8)
#     if i == 1:
#         plt.ylabel('RHC', fontsize = 8)
#
# plt.show()


#####################

# OOD testing

#####################

# OOD data

# simulate OOD test data
X_OOD1 = np.random.rand(1000, X_train.shape[1])
X_OOD2 = np.random.choice([-1, 1], size=[1000,X_train.shape[1]], p=[.5, .5])
X_OOD = X_OOD1 + X_OOD2

## LR prediction for OOD testing set
z = np.sum(w * X_OOD, axis = 1) + w0
k = np.clip(z,-100,100)
h = 1/(1+np.exp(-k))
lr_test = h
y_OOD = np.round(lr_test)


test_loader_x = torch.tensor(X_OOD).float()
test_loader_y = torch.tensor(y_OOD).float()
test_loader = DataLoader(TensorDataset(test_loader_x, test_loader_y))

# test on other baselines

#random forest
y_rf = regr.predict(X_OOD)
y_rfround = np.round(y_rf)

print('The OOD consistency of random forest:', round(sum(y_rfround == y_OOD)/len(y_OOD)*100,3),'%')

RF_OODC = round(sum(y_rfround == y_OOD)/len(y_OOD)*100,3)

# kernel ridge regression

y_krl = clf1.predict(X_OOD)
y_krlround = np.round(y_krl)
y_krlround[y_krlround < 0] = 0
y_krlround[y_krlround > 1] = 1


print('The OOD consistency of linear kernel model:', round(sum(y_krlround == y_OOD)/len(y_OOD)*100,3),'%')

LK_OODC = round(sum(y_krlround == y_OOD)/len(y_OOD)*100,3)

y_krp = clf2.predict(X_OOD)
y_krpround = np.round(y_krp)
y_krpround[y_krpround < 0] = 0
y_krpround[y_krpround > 1] = 1

print('The OOD consistency of polynomial kernel model:', round(sum(y_krpround == y_OOD)/len(y_OOD)*100,3),'%')

PK_OODC = round(sum(y_krpround == y_OOD)/len(y_OOD)*100,3)

# test on neural networks
print('Consistency of regularised neural network:')
test_accuracy(modelRS)

BCE_OODC = test_accuracy(modelRS)

print('Consistency of not regularised neural network:')
test_accuracy(modelNR)

NN_OODC = test_accuracy(modelNR)

print('Consistency of L2 regularised neural network:')
test_accuracy(modelL2)

L2_OODC = test_accuracy(modelL2)


lr_sample = np.zeros(len(lr_test))
for i in range(len(lr_test)):
    lr_sample[i] = np.random.choice([0,1], p = [1-lr_test[i], lr_test[i]])

y_OOD = lr_sample

test_loader_x = torch.tensor(X_OOD).float()
test_loader_y = torch.tensor(y_OOD).float()
test_loader = DataLoader(TensorDataset(test_loader_x, test_loader_y))

## LR prediction for training set
z = np.sum(w * X_OOD, axis = 1) + w0
k = np.clip(z,-100,100)
h = 1/(1+np.exp(-k))
lr_test = h
lr_round = np.round(lr_test)

print('The accuracy of logistic regression:', round(sum(lr_round == y_OOD)/len(y_OOD)*100,3),'%')

LR_OODA = round(sum(lr_round == y_OOD)/len(y_OOD)*100,3)

#random forest
y_rf = regr.predict(X_OOD)
y_rfround = np.round(y_rf)

print('The OOD accuracy of random forest:', round(sum(y_rfround == y_OOD)/len(y_OOD)*100,3),'%')

RF_OODA = round(sum(y_rfround == y_OOD)/len(y_OOD)*100,3)

# kernel ridge regression

y_krl = clf1.predict(X_OOD)
y_krlround = np.round(y_krl)
y_krlround[y_krlround < 0] = 0
y_krlround[y_krlround > 1] = 1

print('The OOD accuracy of linear kernel model:', round(sum(y_krlround == y_OOD)/len(y_OOD)*100,3),'%')

LK_OODA = round(sum(y_krlround == y_OOD)/len(y_OOD)*100,3)

y_krp = clf2.predict(X_OOD)
y_krpround = np.round(y_krp)
y_krpround[y_krpround < 0] = 0
y_krpround[y_krpround > 1] = 1

print('The OOD accuracy of polynomial kernel model:', round(sum(y_krpround == y_OOD)/len(y_OOD)*100,3),'%')

PK_OODA = round(sum(y_krpround == y_OOD)/len(y_OOD)*100,3)

# test on neural networks
print('Accuracy of regularised neural network:')
test_accuracy(modelRS)

BCE_OODA = test_accuracy(modelRS)


print('Accuracy of not regularised neural network:')
test_accuracy(modelNR)

NN_OODA = test_accuracy(modelNR)

print('Accuracy of L2 regularised neural network:')
test_accuracy(modelL2)

L2_OODA = test_accuracy(modelL2)

LR_A.append(LR_OODA)
RF_A.append(RF_OODA)
LK_A.append(LK_OODA)
PK_A.append(PK_OODA)
NN_A.append(NN_OODA)
L2_A.append(L2_OODA)
BCE_A.append(BCE_OODA)

RF_C.append(RF_OODC)
LK_C.append(LK_OODC)
PK_C.append(PK_OODC)
NN_C.append(NN_OODC)
L2_C.append(L2_OODC)
BCE_C.append(BCE_OODC)

# LR_A = []
# RF_A = []
# LK_A = []
# PK_A = []
# NN_A = []
# L2_A = []
# BCE_A = []
#
# RF_C = []
# LK_C = []
# PK_C = []
# NN_C = []
# L2_C = []
# BCE_C = []
