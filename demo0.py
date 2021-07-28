'''
This is for Table 1. And figure 4.
'''

import Toy_experiment as exp
from sklearn import ensemble

if __name__ == '__main__':
    # Number of training and test set, 300, 424 and 1000 are available.
    train_num = 424
    test_num = 5000
    # DO NOT change the following parameters.
    integers2one_hot = True
    data_augmentation = True
    model = ensemble.RandomForestRegressor(n_estimators=230)
    method = 'Random_Forest'

    metrics = exp.get_toy_metrics(train_num)
    print('----------------------train---------------------')
    X, y, _ = exp.get_toy_data(metrics, create_more_metrics=data_augmentation, integers2one_hot=integers2one_hot)

    print('----------------------test----------------------')
    # You could change the type='random_test' to resample test data.
    test_metrics = exp.get_toy_metrics(test_num, type='fixed_test', train_num=train_num)
    testX, testy, num_new_metrics = exp.get_toy_data(test_metrics, create_more_metrics=False,
                                                     integers2one_hot=integers2one_hot)

    # If the range=11, you can see the results of Gaussian Process and MLP which are not mentioned in the paper.
    # range(4, 5) is random forest.
    # If you want to see the figure, set the show_fig=True
    exp.try_different_method(X, y, testX, testy, model, method, show_fig=True)