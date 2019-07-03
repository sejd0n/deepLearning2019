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
    file_path = 'data/n2he-_linear_a1.csv'
    true_data_epochs = 100
    timer=0
    timer2 = 0;
    MSEr = [0,0,0,0,0]
    MSElj = [0,0,0,0,0]
    sMAPEr = [0,0,0,0,0]
    sMAPElj = [0,0,0,0,0]
    dataloader_params = {
        'batch_size': 3,
        'shuffle': False,
        'num_workers': 3
    }

    predictions = []


   

    for i in range(runs):
 

        realTrainLoader, realTestLoader = sf.realset_generator3(file_path, 0.5, dataloader_params, lower_boundary, upper_boundary)
     
        print(realTrainLoader, realTestLoader)
        #scaling factor has to be set manually in supporting_functions.py !!!
        scale_factor = sf.get_scaling_factor(realTrainLoader, realTestLoader)
        scaled_real_trainLoader = sf.scale(realTrainLoader, scale_factor)
        scaled_real_testloader = sf.scale(realTestLoader, scale_factor)
     

        print("scaled factor: {}".format(scale_factor))

     
        for s in range(3):
            for a in range(3):
                for n in range(5):
                    for m in range(5):
                        timer2 = time.time()
                        for p in range(1):
                            layers = (n,m)
                            clf = MLPRegressor(solver=solvers[s], activation=activations[a], alpha=0.01, hidden_layer_sizes = layers, learning_rate_init=0.01,  random_state = 1, max_iter=max_iteration)
                            print(clf)
                            tr = len(scaled_real_trainLoader.list_IDs)
                            tro = len(scaled_real_trainLoader.labels)

                            x_val = []
                            y_val = []
                            for i in range(tr):
                                for t in range(3):
                                    x_ite = []
                                    for w in range(3):
                                        x_ite.append(float(scaled_real_trainLoader.list_IDs[i][t][w]))
                                    x_val.append(x_ite)

                            print(x_val)
                            for i in range(tro):
                                for t in range(3):
                                    y_val.append(float(scaled_real_trainLoader.labels[i][t]))

                         
                            clf.partial_fit(x_val, y_val)

                            print("PPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPP")
    
                            timer = time.time()
                            MSE=10
                            num = 0
                            #trainig
                            while(MSE > 0.01 and num<1000):
                                for q in range(20):
                                    clf.partial_fit(x_val, y_val)
                                predictions = clf.predict(x_val)
                                MSE = sf.MSE(predictions, y_val)
                                num +=1
                                print(num, MSE)
                            print("Training", time.time()-timer, "s ", MSE)

                            x_val_sorted = []
                            y_val_sorted = []
                            indx=0;
                            for i in range(len(x_val)):
                                mini = 50
                                mini2= 50
                                indx = 0;
                                for r in range(len(x_val)):
                                    if(x_val[r][0] < mini):
                                        mini=x_val[r][0]
                                        mini2=x_val[r][1]
                                        indx=r
                                    if(x_val[r][0] == mini and x_val[r][1]<mini2):
                                        mini=x_val[r][0]
                                        mini2=x_val[r][1]
                                        indx=r
                                    
                                x_val_sorted.append(x_val.pop(indx))
                                y_val_sorted.append(y_val.pop(indx))

                            predictions = clf.predict(x_val_sorted)
                            for i in range(len(x_val_sorted)):
                                print(x_val_sorted[i][0], x_val_sorted[i][1],x_val_sorted[i][2], y_val_sorted[i], predictions[i])


                            tr = len(scaled_real_testloader.list_IDs)
                            tro = len(scaled_real_testloader.labels)

                            x_test_val = []
                            y_test_val = []
                            for i in range(tr):
                                for t in range(3):
                                    x_ite = []
                                    for w in range(3):
                                        x_ite.append(float(scaled_real_testloader.list_IDs[i][t][w]))
                                    x_test_val.append(x_ite)

                            for i in range(tro):
                                for t in range(3):
                                    y_test_val.append(float(scaled_real_testloader.labels[i][t]))

                            x1=0.7
                            z1=0.7
                            listxz = []
                            temp = []
                            print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
                            for i1 in range(20):                                
                                for i1 in range(20):
                                    temp = []
                                    listxz = []
                                    temp.append(x1)
                                    temp.append(z1)
                                    temp.append(x1+z1)
                                    listxz.append(temp)
                                    print(clf.predict(listxz)[0])
                                    z1+=0.2
                                x1+=0.2
                                print("=======================================")
                              
                            print("&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&")

                                  
                         
                            test_predictions = clf.predict(x_test_val)
                           
                            MSEr[p] = sf.MSE(test_predictions, y_test_val)
                     
                            sMAPEr[p] = sf.SMAPE(test_predictions, y_test_val)
                       
                            for j in range(len(x_test_val)):
                                print(x_test_val[j], test_predictions[j] )
                            print("//////////////////////////")
                         
                            print("MSE: ", MSEr[p])
                            print("SMAPE: ", sMAPEr[p])
                 
                            if p==4:
                                print("****************************************************************************************************")
                                Results.append([solvers[s],activations[a], hidden_neurons[n], hidden_neurons[m], sum(MSEr)/len(MSEr), sum(sMAPEr)/len(sMAPEr), sum(MSElj)/len(MSElj),(time.time()-timer2)/5])
                                print(solvers[s],activations[a], hidden_neurons[n], hidden_neurons[m], sum(MSEr)/len(MSEr), sum(sMAPEr)/len(sMAPEr),(time.time()-timer2)/5)

    print("-------------------------------------------------------------------------------------------------------------")
    print(Results)
