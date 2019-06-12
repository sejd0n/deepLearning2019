import supporting_functions as sf

import time
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
    max_iteration = 200000
    learning_rate = 0.01
    num = 0;
    MSE = 10;
    Results = [];
    x_test = [0.9, 0.95, 1, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6 ,1.7, 1.8, 1.9, 2.0, 2.1, 2.2, 2.3, 2.4, 2.5, 5, 10, 50]
    solvers = [ "sgd","adam", "lbfgs"]
    activations = ["logistic", "relu", "tanh"]
    hidden_neurons = [50,100,150,200,250]
    x_values = sf.uniform_random_n(lower_boundary, upper_boundary, 350)
    file_path = 'data/sigmau-.csv'
    true_data_epochs = 100
    timer=0
    timer2 = 0;
    MSEr = [0,0,0,0,0]
    MSElj = [0,0,0,0,0]
    sMAPEr = [0,0,0,0,0]
    sMAPElj = [0,0,0,0,0]
    dataloader_params = {
        'batch_size': 1,
        'shuffle': False,
        'num_workers': 1
    }

    predictions = []


   

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
     
        for s in range(1):
            for a in range(1):
                for n in range(5):
                    for m in range(5):
                        timer2 = time.time()
                        for p in range(5):
                            layers = (hidden_neurons[n],hidden_neurons[m])
                            clf = MLPRegressor(solver=solvers[s], activation=activations[a], alpha=0.01, hidden_layer_sizes = layers, learning_rate_init=0.01,  random_state = 1, max_iter=max_iteration)
                            print(clf)
                            
                            real_x_train_values = torch.tensor(scaled_real_trainLoader.list_IDs).reshape(-1,1)
                            real_y_train_values = torch.tensor(scaled_real_trainLoader.labels).reshape(-1,1)

                           # print(real_y_train_values)
                            timer = time.time()
                            while(MSE > 0.05 and num < 2000):
                                for q in range(50):
                                    clf.partial_fit(x_values.reshape(-1,1), sf.get_LJ(x_values).reshape(-1,1))
                                predictions = clf.predict(x_values.reshape(-1,1))
                                listes = sf.get_LJ_list(x_values.reshape(-1,1))
                                MSE = sf.MSE(predictions, listes)
                                num += +1
                                print(num, MSE)
                            for x in range(len(predictions)):
                                print(float(x_values.reshape(-1,1)[x]), predictions[x], float(listes[x]), predictions[x]-float(listes[x]))
                            
                                
                            print("Pretraining", time.time()-timer, "s ", MSE)
                            timer = time.time()
                            MSE=10
                            num = 0
                            #time.sleep(30)
                            while(MSE > 0.01 and num<5000):
                                for q in range(20):
                                    clf.partial_fit(real_x_train_values, real_y_train_values)
                                predictions = clf.predict(real_x_train_values)
                                MSE = sf.MSE(predictions, real_y_train_values)
                                num +=1
                                print(num, MSE)
                            print("Training", time.time()-timer, "s ", MSE)
                            
                            test_x_values = torch.tensor(scaled_real_testloader.list_IDs).detach().numpy()
                            predictions = clf.predict(test_x_values.reshape(-1,1))
                           # print("TRAINING COMPLETE")
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
                            MSEr[p] = sf.MSE(sorted_predictions, sorted_true_y_values)
                            MSElj[p] = sf.MSE(sf.get_LJ_list(sorted_x_values), sorted_true_y_values)
                            sMAPEr[p] = sf.SMAPE(sorted_predictions, sorted_true_y_values)
                            sMAPElj[p] = sf.SMAPE(sf.get_LJ_list(sorted_x_values), sorted_true_y_values)
                            for j in x_test:
                                print(j, float(clf.predict(j)), float(sf.get_LJ(j)))
                            print("//////////////////////////")
                            print(list(zip(sorted_x_values, sorted_predictions, sorted_true_y_values)))
                            print("MSE: ", MSEr[p])
                            print("SMAPE: ", sMAPEr[p])
                            print("MSE LJ: ", MSElj[p])
                            print("SMAPE LJ: ", sMAPElj[p])
                            if p==4:
                                print("****************************************************************************************************")
                                #print(clf)
                                #print("MSE", sum(MSEr)/len(MSEr))
                                #print("sMAPE", sum(sMAPEr)/len(sMAPEr))
                                #print("MSE", sum(MSElj)/len(MSElj))
                                #print("sMAPE", sum(sMAPElj)/len(sMAPElj))
                                #print("AVERAGE TIME:", (time.time()-timer2)/5)
                                Results.append([solvers[s],activations[a], hidden_neurons[n], hidden_neurons[m], sum(MSEr)/len(MSEr), sum(sMAPEr)/len(sMAPEr), sum(MSElj)/len(MSElj),(time.time()-timer2)/5])
                                print(solvers[s],activations[a], hidden_neurons[n], hidden_neurons[m], sum(MSEr)/len(MSEr), sum(sMAPEr)/len(sMAPEr), sum(MSElj)/len(MSElj),(time.time()-timer2)/5)

                            #plt.plot(sorted_x_values, sorted_predictions, label="predicted values")
                            #plt.plot(sorted_x_values, sorted_true_y_values, label="true data")
                            #plt.plot(sorted_x_values, sf.get_generalized_approximation(numpy.array(sorted_x_values)), label="generalized function")
                            #plt.plot(sorted_x_values, )
                            #plt.xlabel("Range [A]")
                            #plt.ylabel("force deviation from mean [eV]")
                            #plt.legend()
                            #plt.show()
    print("-------------------------------------------------------------------------------------------------------------")
    print(Results)
