from argument import args
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from data_utils import load_data
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import TensorDataset, DataLoader

def svm(kernel = args.svm_kernel):
    model = SVC(kernel=kernel)
    train_x, train_y, val_x, val_y, test_x, test_y = load_data(args.data_name)

    model.fit(train_x, train_y)

    train_pred = model.predict(train_x)
    val_pred = model.predict(val_x)
    test_pred = model.predict(test_x)

    train_acc = accuracy_score(train_pred, train_y)
    val_acc = accuracy_score(val_pred, val_y)
    test_acc = accuracy_score(test_pred, test_y)

    print(f'{args.svm_kernel} kernel SVM learning result\n')
    print(f'train_acc: {train_acc}, val_acc: {val_acc}, test_acc: {test_acc} ')

def rogistic_regression():
    #GPU를 사용할 수 있을 경우 사용합니다.
    USE_CUDA = torch.cuda.is_available()
    device = torch.device("cuda" if USE_CUDA else "cpu")

    train_x, train_y, val_x, val_y, test_x, test_y = load_data(args.data_name)

    training_epochs = args.regression_epoch
    batch_size = args.regression_batch
    min_cost = 1000
    val_cost_list = []
    train_cost_list = []
    dataset = TensorDataset(torch.FloatTensor(train_x), torch.LongTensor(train_y))
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    model = nn.Linear(train_x.shape[-1],len(np.unique(train_y)), bias=True).to(device)
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.regression_learning_rate)
    print('Regression Learning start\n')
    for epoch in range(training_epochs):  # 앞서 training_epochs의 값은 15로 지정함.
        avg_cost = 0
        total_batch = len(data_loader)

        for X, Y in data_loader:
            X = X.to(device)
            Y = Y.to(device)

            optimizer.zero_grad()
            hypothesis = model(X)
            cost = criterion(hypothesis, Y)
            cost.backward()
            optimizer.step()

            avg_cost += cost / total_batch
        train_cost_list.append(avg_cost.item())
        with torch.no_grad():
            X_val = torch.FloatTensor(val_x).to(device)
            Y_val = torch.LongTensor(val_y).to(device)
            val_hypothesis = model(X_val)
            val_cost = criterion(val_hypothesis, Y_val)

            if val_cost <= min_cost:
                min_cost = val_cost
                torch.save(model.state_dict(), 'regression_model')
        val_cost_list.append(val_cost.item())
        print('Epoch:', '%04d' % (epoch + 1), 'cost =', '{:.9f}'.format(avg_cost), 'val_cost:',
              '{:.9f}'.format(val_cost.item()))


    model.load_state_dict(torch.load('regression_model'))
    with torch.no_grad():  # torch.no_grad()를 하면 gradient 계산을 수행하지 않는다.

        X_train = torch.FloatTensor(train_x).to(device)
        Y_train = torch.LongTensor(train_y).to(device)
        train_prediction = model(X_train)
        train_prediction = torch.argmax(train_prediction, 1) == Y_train
        train_acc = train_prediction.float().mean()

        X_val = torch.FloatTensor(val_x).to(device)
        Y_val = torch.LongTensor(val_y).to(device)
        val_prediction = model(X_val)
        val_prediction = torch.argmax(val_prediction, 1) == Y_val
        val_acc = val_prediction.float().mean()

        X_test = torch.FloatTensor(test_x).to(device)
        Y_test = torch.LongTensor(test_y).to(device)
        test_prediction = model(X_test)
        test_prediction = torch.argmax(test_prediction, 1) == Y_test
        test_acc = test_prediction.float().mean()

        print(f'Logistic regression learning result\n')
        print(f'train_acc: {train_acc}, val_acc: {val_acc}, test_acc: {test_acc} ')

        plt.plot([i for i in range(len(val_cost_list))], train_cost_list)
        plt.plot([i for i in range(len(val_cost_list))], val_cost_list)
        plt.title(f'Logistic regression learning graph')
        plt.legend(['train_loss', 'val_loss'])
        plt.xlabel('epoch')
        plt.ylabel('loss')
        plt.show()

def mlp():
    USE_CUDA = torch.cuda.is_available()
    device = torch.device("cuda" if USE_CUDA else "cpu")
    training_epochs = args.mlp_epoch
    batch_size = args.mlp_batch
    min_cost = 1000
    val_cost_list = []
    train_cost_list = []
    train_x, train_y, val_x, val_y, test_x, test_y = load_data(args.data_name)
    dataset = TensorDataset(torch.FloatTensor(train_x), torch.LongTensor(train_y))
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    model = nn.Sequential()
    model.add_module('fc1', nn.Linear(train_x.shape[-1], 30))
    model.add_module('relu1', nn.ReLU())
    model.add_module('fc2', nn.Linear(30, 30))
    model.add_module('relu2', nn.ReLU())
    model.add_module('fc3', nn.Linear(30, 30))
    model.add_module('relu3', nn.ReLU())
    model.add_module('fc4', nn.Linear(30,len(np.unique(train_y))))
    model.to(device)

    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.mlp_learning_rate)
    print('MLP Learning start\n')
    for epoch in range(training_epochs):
        avg_cost = 0
        total_batch = len(data_loader)

        for X, Y in data_loader:

            X = X.to(device)
            Y = Y.to(device)

            optimizer.zero_grad()
            hypothesis = model(X)
            cost = criterion(hypothesis, Y)
            cost.backward()
            optimizer.step()

            avg_cost += cost / total_batch
        train_cost_list.append(avg_cost.item())
        with torch.no_grad():
            X_val = torch.FloatTensor(val_x).to(device)
            Y_val = torch.LongTensor(val_y).to(device)
            val_hypothesis = model(X_val)
            val_cost = criterion(val_hypothesis, Y_val)

            if val_cost <= min_cost:
                min_cost = val_cost
                torch.save(model.state_dict(), 'mlp_model')
        val_cost_list.append(val_cost.item())
        print('Epoch:', '%04d' % (epoch + 1), 'cost =', '{:.9f}'.format(avg_cost), 'val_cost:',
              '{:.9f}'.format(val_cost.item()))

    model.load_state_dict(torch.load('mlp_model'))
    with torch.no_grad():  # torch.no_grad()를 하면 gradient 계산을 수행하지 않는다.
        X_train = torch.FloatTensor(train_x).to(device)
        Y_train = torch.LongTensor(train_y).to(device)
        train_prediction = model(X_train)
        train_prediction = torch.argmax(train_prediction, 1) == Y_train
        train_acc = train_prediction.float().mean()

        X_val = torch.FloatTensor(val_x).to(device)
        Y_val = torch.LongTensor(val_y).to(device)
        val_prediction = model(X_val)
        val_prediction = torch.argmax(val_prediction, 1) == Y_val
        val_acc = val_prediction.float().mean()

        X_test = torch.FloatTensor(test_x).to(device)
        Y_test = torch.LongTensor(test_y).to(device)
        test_prediction = model(X_test)
        test_prediction = torch.argmax(test_prediction, 1) == Y_test
        test_acc = test_prediction.float().mean()

        print(f'mlp regression learning result\n')
        print(f'train_acc: {train_acc}, val_acc: {val_acc}, test_acc: {test_acc} ')

    plt.plot([i for i in range(len(val_cost_list))], train_cost_list)
    plt.plot([i for i in range(len(val_cost_list))], val_cost_list)
    plt.title(f'MLP learning grph')
    plt.legend(['train_loss', 'val_loss'])
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.show()


