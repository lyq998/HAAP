from nasbench import api
import numpy as np
import random
import os
import pickle

# Replace this string with the path to the downloaded nasbench.tfrecord before
# executing.
NASBENCH_TFRECORD = os.path.join('path', 'nasbench_only108.tfrecord')
NASBENCH_MAX_LEN = 423624

INPUT = 'input'
OUTPUT = 'output'
CONV1X1 = 'conv1x1-bn-relu'
CONV3X3 = 'conv3x3-bn-relu'
MAXPOOL3X3 = 'maxpool3x3'
NULL = 'null'


# get data randomly
def get_data_index_from_101(num, type='train', train_num=500):
    # type is for 'train', 'fixed_test' or 'random_test'
    # train_num is loading the train indexes and keep them out of the test
    # if the type is not 'train', be sure the train_index_num is correct
    if not os.path.isdir('pkl'):
        os.makedirs('pkl')
    if type in ['train', 'fixed_test']:
        if type == 'fixed_test':
            # the first number in fixed_test_data represents the number of the train data
            # (which will be removed and will not appear in test_data),
            # the second number is the number of test data
            save_path = os.path.join('pkl', 'fixed_test_data{:0>6d}_{:0>6d}.pkl'.format(train_num, num))
        else:
            save_path = os.path.join('pkl', 'train_data{:0>6d}.pkl'.format(num))
        if os.path.isfile(save_path):
            with open(save_path, 'rb') as file:
                random_list = pickle.load(file)
            print('Exist {:s}_data.pkl, loading...'.format(type))
            # print('The indexes of {:s} architecture: {:}'.format(type, random_list))
        else:
            # sample
            max_number = NASBENCH_MAX_LEN
            list_remove_train = list(range(0, max_number))

            if type in ['fixed_test', 'random_test']:
                train_data_index_path = os.path.join('pkl', 'train_data{:0>6d}.pkl'.format(train_num))
                print('Removing train list (len: {:}) to sample...'.format(train_num))
                with open(train_data_index_path, 'rb') as file:
                    train_list = pickle.load(file)
                for i in range(train_num):
                    list_remove_train.remove(train_list[i])

            print('left: {:}'.format(len(list_remove_train)))
            random_list = random.sample(list_remove_train, num)
            random_list.sort()
            with open(save_path, 'wb') as file:
                pickle.dump(random_list, file)
            print('Run for the first time! Create new {:s}_data.pkl'.format(type))
            # print('The indexes of {:s} architecture: {:}'.format(type, random_list))
    else:
        # 'random_test' don't save and load
        max_number = NASBENCH_MAX_LEN
        random_list = random.sample(range(0, max_number), num)
        random_list.sort()

    return random_list


def get_MAX():
    # get max trainable number and max full training time
    nasbench = api.NASBench(NASBENCH_TFRECORD)
    max_full_training_time = 0
    max_trainable_parameters = 0
    for unique_hash in nasbench.hash_iterator():
        fixed_metrics, computed_metrics = nasbench.get_metrics_from_hash(unique_hash)
        final_training_time_list = []
        for i in range(3):
            # the 108 means the train epochs, and I have only downloaded the metrics in Epoch 108 (full train)
            # the three iterations: three results of independent experiments recorded in the dataset
            final_training_time_list.append(computed_metrics[108][i]['final_training_time'])
        # use the mean of three metrics
        final_training_time = np.mean(final_training_time_list)
        if final_training_time > max_full_training_time:
            max_full_training_time = final_training_time
        trainable_parameters = fixed_metrics['trainable_parameters']
        if trainable_parameters > max_trainable_parameters:
            max_trainable_parameters = trainable_parameters
    return max_full_training_time, max_trainable_parameters


def get_corresponding_metrics_by_index(index_list, type='train'):
    # type is for 'train', 'fixed_test' or 'random_test'
    # 'random_test' for not saving and loading
    iter_num = 0
    # use the len of index_list and the first index to distinguish different index_list
    save_path = os.path.join('pkl', '{:s}_metrics{:0>6d}_{:0>6d}.pkl'.format(type, len(index_list), index_list[0]))
    print('\nGetting the corresponding metrics by index.')
    if not os.path.isfile(save_path):
        nasbench = api.NASBench(NASBENCH_TFRECORD)
        important_metrics = {}
        for unique_hash in nasbench.hash_iterator():
            iter_num += 1

            if iter_num in index_list:
                fixed_metrics, computed_metrics = nasbench.get_metrics_from_hash(unique_hash)
                final_training_time_list = []
                final_test_accuracy_list = []
                for i in range(3):
                    # the 108 means the train epochs, and I have only downloaded the metrics in Epoch 108 (full train)
                    # the three iterations: three results of independent experiments recorded in the dataset
                    final_training_time_list.append(computed_metrics[108][i]['final_training_time'])
                    final_test_accuracy_list.append(computed_metrics[108][i]['final_test_accuracy'])
                # use the mean of three metrics
                final_training_time = np.mean(final_training_time_list)
                final_test_accuracy = np.mean(final_test_accuracy_list)
                # print('final training time: {:}'.format(final_training_time))
                # print('fianl test accuracy: {:}'.format(final_test_accuracy))

                # using the index to create dicts
                important_metrics[iter_num] = {}
                important_metrics[iter_num]['fixed_metrics'] = fixed_metrics
                important_metrics[iter_num]['final_training_time'] = final_training_time
                important_metrics[iter_num]['final_test_accuracy'] = final_test_accuracy

        # print(len(important_metrics))
        if type in ['train', 'fixed_test']:
            # don't save 'random_test'
            with open(save_path, 'wb') as file:
                pickle.dump(important_metrics, file)
    else:
        with open(save_path, 'rb') as file:
            important_metrics = pickle.load(file)
        print('Loading: {:}'.format(save_path))
    return important_metrics


# this function is for padding zero for matrix which is not 7*7
# zeros will be added at penultimate row (or column)
# input: important_metrics
# output: the metrics after padding
def padding_zero_in_matrix(important_metrics):
    for i in important_metrics:
        len_operations = len(important_metrics[i]['fixed_metrics']['module_operations'])
        if len_operations != 7:
            # if the operations is less than 7
            for j in range(len_operations, 7):
                important_metrics[i]['fixed_metrics']['module_operations'].insert(j - 1, 'null')
            # print(important_metrics[i]['fixed_metrics']['module_operations'])

            adjecent_matrix = important_metrics[i]['fixed_metrics']['module_adjacency']
            padding_matrix = np.insert(adjecent_matrix, len_operations - 1,
                                       np.zeros([7 - len_operations, len_operations]), axis=0)
            padding_matrix = np.insert(padding_matrix, [len_operations - 1], np.zeros([7, 7 - len_operations]), axis=1)
            important_metrics[i]['fixed_metrics']['module_adjacency'] = padding_matrix
    return important_metrics


# transform the operations list to integers list
# input: important_metrics
# output: the metrics after padding
def operations2integers(important_metrics):
    dict_oper2int = {NULL: 0, CONV1X1: 1, CONV3X3: 2, MAXPOOL3X3: 3}
    for i in important_metrics:
        fix_metrics = important_metrics[i]['fixed_metrics']
        module_operations = fix_metrics['module_operations']
        module_integers = np.array([dict_oper2int[x] for x in module_operations[1: -1]])
        # use [1: -1] to remove 'input' and 'output'
        important_metrics[i]['fixed_metrics']['module_integers'] = module_integers
    return important_metrics


# delete the first column and the last row in the matrix (because they are zeros)
# input: adjacent matrix (7*7)
# output: adjacent matrix (6*6)
def delete_margin(matrix):
    return matrix[:-1, 1:]


if __name__ == '__main__':
    max_time, max_parameters = get_MAX()
    print(f'The max full training time is: {max_time}, the max number of parameters is: {max_parameters}')
    # The max full training time is: 5521.803059895833, the max number of parameters is: 49979274
