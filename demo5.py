'''
This is for Figure 7.
'''
import pickle
import os
import numpy as np
import matplotlib.pyplot as plt
from Toy_experiment import get_toy_metrics, get_toy_data, try_different_method, model, method


def draw_plot(All_KTau_list, All_MSE_list):
    color = ['hotpink', 'skyblue', 'darksalmon']
    plt.figure(figsize=(5, 5))
    for i, KTau_list in enumerate(All_KTau_list):
        plt.plot(np.arange(0, 121, 10), KTau_list, color[i], linestyle=':', marker='^', label='Random{}'.format(i))
    plt.legend(loc="best")
    plt.xlabel('Number of Augmentation Data')
    plt.ylabel('KTau')
    plt.show()

    plt.figure(figsize=(5, 5))
    for i, MSE_list in enumerate(All_MSE_list):
        plt.plot(np.arange(0, 121, 10), MSE_list, color[i], linestyle=':', marker='D', label='Random{}'.format(i))
    plt.legend(loc="best")
    plt.xlabel('Number of Augmentation Data')
    plt.ylabel('MSE')
    plt.show()


def train_process(train_num, test_num, integers2one_hot, more_train_data, repeat_times):
    save_path = r'pkl/num_creations.pkl'
    if os.path.exists(save_path):
        print('Existing num_creations.pkl')
        with open(save_path, 'rb') as file:
            dic = pickle.load(file)
        All_KTau_list = dic['All_KTau_list']
        All_MSE_list = dic['All_MSE_list']

    else:
        All_KTau_list = []
        All_MSE_list = []
        for _ in range(repeat_times):
            KTau_list = []
            MSE_list = []

            metrics = get_toy_metrics(train_num)
            print('----------------------train---------------------')
            X, y, _ = get_toy_data(metrics, create_more_metrics=False, select_upper_tri=False,
                                   integers2one_hot=integers2one_hot)
            print('----------------------test----------------------')
            test_metrics = get_toy_metrics(test_num, type='fixed_test', train_num=train_num)
            testX, testy, num_new_metrics = get_toy_data(test_metrics, create_more_metrics=False,
                                                         select_upper_tri=False,
                                                         integers2one_hot=integers2one_hot)
            # [4] is random forest
            KTau, MSE = try_different_method(X, y, testX, testy, model[4], method[4],
                                             show_fig=False, return_flag=True)
            KTau_list.append(KTau)
            MSE_list.append(MSE)

            for create_num in range(10, 121, 10):
                metrics = get_toy_metrics(train_num)
                print('----------------------train---------------------')
                X, y, _ = get_toy_data(metrics, create_more_metrics=more_train_data, select_upper_tri=False,
                                       integers2one_hot=integers2one_hot,
                                       max_creation=create_num)

                print('----------------------test----------------------')
                test_metrics = get_toy_metrics(test_num, type='fixed_test', train_num=train_num)
                testX, testy, num_new_metrics = get_toy_data(test_metrics, select_upper_tri=False,
                                                             integers2one_hot=integers2one_hot)

                # [4] is random forest
                KTau, MSE = try_different_method(X, y, testX, testy, model[4], method[4],
                                                 show_fig=False, return_flag=True)
                KTau_list.append(KTau)
                MSE_list.append(MSE)

            All_KTau_list.append(KTau_list)
            All_MSE_list.append(MSE_list)

        save_dic = {'All_KTau_list': All_KTau_list, 'All_MSE_list': All_MSE_list}
        with open(save_path, 'wb') as file:
            pickle.dump(save_dic, file)
    return All_KTau_list, All_MSE_list


if __name__ == '__main__':
    train_num = 424
    test_num = 5000
    integers2one_hot = True
    more_train_data = True
    repeat_times = 3

    All_KTau_list, All_MSE_list = train_process(train_num, test_num, integers2one_hot, more_train_data, repeat_times)
    draw_plot(All_KTau_list, All_MSE_list)
