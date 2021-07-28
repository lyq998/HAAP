'''
This is for Section Experiment on Classical Regression Model
'''
import Toy_experiment as exp

if __name__ == '__main__':
    # Number of training and test set, 300, 424 and 1000 are available.
    train_num = 424
    test_num = 5000
    # Change the following parameters to see different cases. The default setting is case 4.
    integers2one_hot = True
    data_augmentation = True

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
    for i in range(9):
        exp.try_different_method(X, y, testX, testy, exp.model[i], exp.method[i], show_fig=False)
