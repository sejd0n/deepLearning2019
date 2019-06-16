import numpy
import torch
import sys
from math import floor
import torch.nn as nn
import csv

from PytorchDataset import PytorchDataset

def get_generalized_approximation(x_values):
    # returns the corresponding y values for the molcule independent approximation function
    # x in this scenario is the distance between the molecules, with y being the force
    # at x==0 the molecules are guaranteed to collapse into a black hole, and no answer exists
    return 4 * 1 * ((1 / x_values) ** 12 - (1 / x_values) ** 6)


def equal_spaced_n(start, stop, n):
    # returns n (rounded up if necessary) equal spaced values between [start, stop>
    return numpy.arange(start, stop, (stop-start)/n)

def uniform_random_n(start, stop, n):
    # returns n uniform randomly picked values between [start,stop>
    return numpy.random.uniform(start, stop, n)


def get_scaling_factor(DataLoader, TestLoader, feature_index=0):
    # returns the factor by which the iterator needs to be scaled to achieve zero weighted scaling
     scale = 0
     distance = 0
    
    
  
     return float(-3031.2149515805877)    
  


def scale(DataLoader, scaling_factor, feature_index=0):
    ids = []
    labels = []
    for i, data in enumerate(DataLoader):
        features, label = data
        #print(features)
        label = label - scaling_factor

        ids.append(features)
        labels.append(label)

    return PytorchDataset(ids, labels)

def descale(DataLoader, scaling_factor, feature_index=0):
    ids = []
    labels = []
    for i, data in enumerate(DataLoader):
        features, label = data
        label = label + scaling_factor

        ids.append(features)
        labels.append(label)

    return PytorchDataset(ids, labels)

def rescale(DataLoader):
    # scales data to be between -1 and 1
    min = sys.maxsize
    max = -sys.maxsize
    for i, item in enumerate(DataLoader):
        value, label = item
        if label<min:
            min = label
        if label>max:
            max = label

    ids = []
    labels = []
    for i, item in enumerate(DataLoader):
        value, label = item
        newlabel = ((label-min)/(max-min)-0.5)*2
        labels.append(newlabel)
        ids.append(value)

    return PytorchDataset(ids, labels)


def numpy_data_to_trainloaders(numpy_data, train_ratio, dataloader_params):
    total_objects = len(numpy_data)
    train_objects = floor(train_ratio * total_objects)
    # validate_objects = floor(validate_ratio * total_objects)

    numpy.random.shuffle(numpy_data)
   # print(numpy_data)
    numpy_train = numpy_data[:int(train_objects)]
    # numpy_validate = numpy_data[train_objects:(train_objects+validate_objects)]
    numpy_test = numpy_data[int(train_objects):]

    partition1_train,partition2_train,partition3_train, labels_train = numpy_to_x_y(numpy_train)
    # partition_validate, labels, validate = nu
    partition1_test,partition2_test,partition3_test, labels_test = numpy_to_x_y(numpy_test)
    
    train_input = numpy.array(list(zip(partition1_train,partition2_train,partition3_train)))
    print(partition1_test[0],partition3_test[0],partition2_test[0])
    print(train_input[0])
    trainset = PytorchDataset(train_input, labels_train)
    trainloader = torch.utils.data.DataLoader(trainset, **dataloader_params)

    # validateset = PytorchDataset(partition_validate, labels_validate)
    # validateloader = torch.utils.data.DataLoader(validateset, **dataloader_params)

    test_input = numpy.array(list(zip(partition1_test,partition2_test,partition3_test)))


    testset = PytorchDataset(test_input, labels_test)
    #print(trainloader)
    testloader = torch.utils.data.DataLoader(testset, **dataloader_params)

    return trainloader, testloader

def datasets_generator(numpy_x, numpy_y, train_ratio, dataloader_params, **kwargs):
    # numpy data should be rows of x, y
    # returns a training and test torch dataset, according to trainig_ratio

    # if train_ratio+validate_ratio>=1:
    #     print("datasets generator got invalid configuration, exiting")
    #     exit()

    numpy_data = numpy.array(list(zip(numpy_x, numpy_y)))

    return numpy_data_to_trainloaders(numpy_data, train_ratio, dataloader_params)


def realset_generator(csv_path, train_ratio, dataloader_params, lower_range, upper_range):
    with open(csv_path) as csv_file:
        csv_reader = csv.DictReader(csv_file, delimiter = " ")
        real_x = []
        real_y = []
        # todo figure out what MRCI is and correct it
        for line in csv_reader:
            distance = float(line['R [A]'])
            if distance >= lower_range and distance <= upper_range:
                real_x.append(distance)
                real_y.append(float(line['MRCI1 [eV]']))
        # print(real_x)

        numpy_data = numpy.array(list(zip(real_x, real_y)))

        return numpy_data_to_trainloaders(numpy_data, train_ratio, dataloader_params)

def realset_generator3(csv_path, train_ratio, dataloader_params, lower_range, upper_range):
    with open(csv_path) as csv_file:
        csv_reader = csv.DictReader(csv_file, delimiter = " ")
        real_x1 = []
        real_x2 = []
        real_x3 = []
        real_y = []
        # todo figure out what MRCI is and correct it
        for line in csv_reader:
            distance = float(line['NNR'])
            distance2 = float(line['HeNR'])
           # if distance >= lower_range and distance <= upper_range:
            x = distance
            y = distance2
            z = x + y
           # print(x, y, z)
            real_x1.append(x)
            real_x2.append(y)
            real_x3.append(z)
            real_y.append(float(line['MRCI1']))
      

        numpy_data = numpy.array(list(zip(real_x1,real_x2,real_x3, real_y)))
        #numpy_data = numpy.dstack(real_x1,real_x2,real_x3, real_y)
        #print(real_x3)
        return numpy_data_to_trainloaders(numpy_data, train_ratio, dataloader_params)


def numpy_to_x_y(numpy):
    #print(numpy)
    #print("JJJJJJJJJJJJJJJJJJJJJJJJJJ")
    return numpy[:,0], numpy[:,1], numpy[:,2], numpy[:,3]

def train_scenario(net, criterion, trainloader, optimizer):


    # use_gpu = torch.cuda.is_available()


    # if use_gpu:
    #     net = net.cuda()


    # optimizer = optim.SGD(net.parameters(), lr=learning_rate, momentum=momentum)


    for i, data in enumerate(trainloader):
        # get the inputs
        inputs, labels = data

        # zero the parameter gradients
        optimizer.zero_grad()

        outputs = net(inputs)

        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

    return net

def train(net, criterion, trainloader, real_trainloader, optimizer):
    trainloader = rescale(trainloader)


    # todo consider initializing to the mean of trainloader. and reinitializing for realtest
    # pretraining on artificial data
    net = train_scenario(net, criterion, trainloader, optimizer=optimizer)

    # criterion = nn.MSELoss()
    # for i in range(true_data_epochs):
    #
    #     sf.train_scenario(Net(), criterion, scaled_real_trainLoader)

    return net

def predict_scenario(net, criterion, testloader):
    total_loss = 0
    all_inputs = []
    for i, data in enumerate(testloader):
        inputs, labels = data
        # print("inputs {}: ".format(inputs))
        predictions = net(inputs)
        all_inputs.extend(inputs)
        # print("predictions: {} labels: {}".format(predictions, labels))
        loss = criterion(predictions, labels)
        total_loss += loss
    # print(i)
    # print("all inputs")
    # print(all_inputs)
    all_predictions = net(torch.tensor(all_inputs))
    return total_loss, i+1, all_predictions

def print_dataset(dataset):
    for i, data in enumerate(dataset):
        features, label = data
       # print(features[0][0])
        print(float(features[0]),float(features[1]),float(features[2]), float(label))

def get_LJ(x):
    return 4*3*((1/x)**12-(1/x)**6)

def get_LJ_list(x_values):
    y_values = []
    for x in x_values:
        y_values.append(get_LJ(x))
    return y_values

def MSE(New, Old):
    sum = 0
    for x in range(len(New)):
        sum += (float(New[x])-float(Old[x]))**2
     #   print(x, sum, float(New[x]), float(Old[x]), float(len(New))) 
    return sum/float(len(New))

def SMAPE(New, Old):
    sum = 0
    for x in range(len(New)):
        sum += abs(Old[x]-New[x])/((New[x]+Old[x])/2)

    return sum/len(New)



