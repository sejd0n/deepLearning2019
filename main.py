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
    solvers = [ "sgd","adam", "lbfgs"]
    activations = ["logistic", "relu", "tanh"]
    hidden_neurons = [50,100,150,200,250]
    x_values = sf.uniform_random_n(lower_boundary, upper_boundary, 350)
    #path to file with training data
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


   
    #Loop for multiple runs
    for i in range(runs):
        #Initial Extract and split data into train and test sets
        realTrainLoader, realTestLoader = sf.realset_generator(file_path, 0.5, dataloader_params, lower_boundary, upper_boundary)

        #Get scaling factor and pront it
        scale_factor = sf.get_scaling_factor(realTrainLoader, realTestLoader)
        scaled_real_trainLoader = sf.scale(realTrainLoader, scale_factor)
        scaled_real_testloader = sf.scale(realTestLoader, scale_factor)
        print("scaled factor: {}".format(scale_factor))

        #print scalet test data
        sf.print_dataset(scaled_real_testloader)

        #Loop for choosing different solvers
        for s in range(3):
            #Loop for choosing different Activation functions
            for a in range(3):
                #Loop for choosing different number of neurons in 1st hidden layer
                for n in range(5):
                    #Loop for choosing different number of neurons in 2nd hidden layer
                    for m in range(5):
                        #Start of time measurment
                        timer2 = time.time()
                        #Loop of multiple runs with the same setting 
                        for p in range(5):

                            #Creating NN and print the setting
                            layers = (hidden_neurons[n],hidden_neurons[m])
                            clf = MLPRegressor(solver=solvers[s], activation=activations[a], alpha=0.01, hidden_layer_sizes = layers, learning_rate_init=0.01,  random_state = 1, max_iter=max_iteration)
                            print(clf)

                            #Extract and split data into train and test sets
                            realTrainLoader, realTestLoader = sf.realset_generator(file_path, 0.5, dataloader_params, lower_boundary, upper_boundary)

                            #reshape data
                            real_x_train_values = torch.tensor(scaled_real_trainLoader.list_IDs).reshape(-1,1)
                            real_y_train_values = torch.tensor(scaled_real_trainLoader.labels).reshape(-1,1)

                           
                            timer = time.time()
                            #Pre-traning
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
                            
                            #Traning
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
                           
                            real_x_values = torch.tensor(scaled_real_testloader.list_IDs).detach().numpy()
                            true_y_values = torch.tensor(scaled_real_testloader.labels).detach().numpy()
                            
                            #Result extraction
                            indexes = numpy.argsort(real_x_values)
                            sorted_x_values = []
                            sorted_predictions = []
                            sorted_true_y_values = []
                            
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

                            #Results averege for all five runs
                            if p==4:
                                print("****************************************************************************************************")
                                Results.append([solvers[s],activations[a], hidden_neurons[n], hidden_neurons[m], sum(MSEr)/len(MSEr), sum(sMAPEr)/len(sMAPEr), sum(MSElj)/len(MSElj),(time.time()-timer2)/5])
                                print(solvers[s],activations[a], hidden_neurons[n], hidden_neurons[m], sum(MSEr)/len(MSEr), sum(sMAPEr)/len(sMAPEr), sum(MSElj)/len(MSElj),(time.time()-timer2)/5)

    #Print final results after all configuration runs
    print("-------------------------------------------------------------------------------------------------------------")
    print(Results)
