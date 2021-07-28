'''
This is for Figure 6.
'''
import Toy_experiment as exp
import pickle

if __name__ == '__main__':
    # If this is True, it will cover the original data and rerun the fine tuning process. It may take some time.
    rerun = False
    # DO NOT change the following parameters.
    # Number of training and test set
    train_num = 424
    test_num = 5000
    # DO NOT change the following parameters.
    integers2one_hot = True
    data_augmentation = True

    if rerun:
        metrics = exp.get_toy_metrics(train_num)
        print('----------------------train---------------------')
        X, y, _ = exp.get_toy_data(metrics, create_more_metrics=data_augmentation, integers2one_hot=integers2one_hot)

        print('----------------------test----------------------')
        # You could change the type='random_test' to resample test data.
        test_metrics = exp.get_toy_metrics(test_num, type='fixed_test', train_num=train_num)
        testX, testy, num_new_metrics = exp.get_toy_data(test_metrics, create_more_metrics=False,
                                                         integers2one_hot=integers2one_hot)

        method_name = 'random_forest'
        exp.Ablation_study(method_name, X, y, testX, testy, [10, 310], step=10)
    else:
        save_path = r'pkl\ktau_and_mse_list.pkl'
        with open(save_path, 'rb') as file:
            load_dic = pickle.load(file)
            print('Load KTau and MSE list successfully!')

        exp.make_plot_for_KTau_and_MSE(load_dic['KTau_list'], load_dic['MSE_list'])
