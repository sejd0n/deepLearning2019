import supporting_functions as sf

import torch
import torch.nn as nn
import numpy
import matplotlib.pyplot as plt
import torch.nn.functional as F
import torch.optim as optim

from sklearn.neural_network import MLPRegressor

# correct for scale difference neural network and expected range

if __name__ == '__main__':
    # freeze_support()
    accuracies = []
    runs = 1
    lower_boundary = 0.85
    upper_boundary = 3
    max_iteration = 10000
    learning_rate = 0.01
    solvers = ["adam", "sgd", "lbfgs"]
    activations = ["relu", "logistic", "tanh"]
    hidden_neurons = [50,100,150,200,250]
    x_values = sf.uniform_random_n(lower_boundary, upper_boundary, 30000)
    file_path = 'data/sigmau-.csv'
    true_data_epochs = 100
    dataloader_params = {
        'batch_size': 1,
        'shuffle': False,
        'num_workers': 1
    }

    predictions = []


    class Net(nn.Module):
        def __init__(self):
            super(Net, self).__init__()

            self.fc1 = nn.Linear(1, 10)
            # self.fc2 = nn.Linear(100, 100)
            self.fc3 = nn.Linear(10, 1)


        def forward(self, x):
            x = x.view(-1, 1)
            x = F.relu(self.fc1(x))
            # x = self.fc2(x)
            x = F.relu(self.fc3(x))
            return x

    for i in range(runs):
        # trainloader, testloader = sf.datasets_generator(x_values, sf.get_generalized_approximation(x_values), 0.8, dataloader_params)
        # trainloader = sf.rescale(trainloader)
        # testloader = sf.rescale(testloader)

        realTrainLoader, realTestLoader = sf.realset_generator(file_path, 0.5, dataloader_params, lower_boundary, upper_boundary)

        scale_factor = sf.get_scaling_factor(realTrainLoader, realTestLoader)
        scaled_real_trainLoader = sf.scale(realTrainLoader, scale_factor)
        scaled_real_testloader = sf.scale(realTestLoader, scale_factor)
        # scaled_real_trainLoader = sf.rescale(scaled_real_trainLoader)
        # scaled_real_testloader = sf.rescale(scaled_real_testloader)
        # criterion = nn.MSELoss()
        # net = Net()
        # optimizer = optim.Adam(net.parameters(), lr=learning_rate)
        # net = sf.train(net, criterion, trainloader, scaled_real_testloader, optimizer)

        print("scaled factor: {}".format(scale_factor))
        sf.print_dataset(scaled_real_testloader)
        #print(scaled_real_testloader)
        # todo i might be doubly dividing the error  by count, ensure  this isn't the case
        # prediction_criterion = nn.MSELoss()
        # loss, count, predictions = sf.predict_scenario(net, criterion, scaled_real_testloader)

        # print("total loss is: {} and there are {} objects".format(loss, count))
        # print("average loss is: {}".format(loss / count))
        # accuracies.append(float(loss / count))
        # print(sf.get_generalized_approximation(x_values).reshape)
        for s in range(3):
            for a in range(3):
                for n in range(5):
                    clf = MLPRegressor(solver=solvers[s], activation=activations[a], alpha=0.01, hidden_layer_sizes = hidden_neurons[n], learning_rate_init=0.01,  random_state = 1, max_iter=max_iteration)
                    print(clf)

                    real_x_train_values = torch.tensor(scaled_real_trainLoader.list_IDs).reshape(-1,1)
                    real_y_train_values = torch.tensor(scaled_real_trainLoader.labels).reshape(-1,1)

                   # print(real_y_train_values)

                    clf.fit(x_values.reshape(-1,1), sf.get_generalized_approximation(x_values).reshape(-1,1))
                    clf.fit(real_x_train_values, real_y_train_values)

                    test_x_values = torch.tensor(scaled_real_testloader.list_IDs).detach().numpy()
                    predictions = clf.predict(test_x_values.reshape(-1,1))
                    print("TRAINING COMPLETE")
                    #print(predictions)
                    # print(predictions)
                    # predictions = predictions.detach().numpy()

                    real_x_values = torch.tensor(scaled_real_testloader.list_IDs).detach().numpy()
                    true_y_values = torch.tensor(scaled_real_testloader.labels).detach().numpy()
                    # print(x_values)

                    indexes = numpy.argsort(real_x_values)
                    sorted_x_values = []
                    sorted_predictions = []
                    sorted_true_y_values = []
                    #print(x_values)
                    for index in indexes:
                        sorted_x_values.append(real_x_values[index])
                        sorted_predictions.append(predictions[index])
                        sorted_true_y_values.append(true_y_values[index])

                    print(list(zip(sorted_x_values, sorted_predictions, sorted_true_y_values)))
                    print("MSE: ", sf.MSE(sorted_predictions, sorted_true_y_values))
                    print("SMAPE: ", sf.SMAPE(sorted_predictions, sorted_true_y_values))
                    print("MSE LJ: ", sf.MSE(sf.get_LJ_list(sorted_x_values), sorted_true_y_values))
                    print("SMAPE LJ: ", sf.SMAPE(sf.get_LJ_list(sorted_x_values), sorted_true_y_values))

                    #plt.plot(sorted_x_values, sorted_predictions, label="predicted values")
                    #plt.plot(sorted_x_values, sorted_true_y_values, label="true data")
                    #plt.plot(sorted_x_values, sf.get_generalized_approximation(numpy.array(sorted_x_values)), label="generalized function")
                    #plt.plot(sorted_x_values, )
                    #plt.xlabel("Range [A]")
                    #plt.ylabel("force deviation from mean [eV]")
                    #plt.legend()
                    #plt.show()
