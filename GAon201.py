from demo3 import get_metrics_from_index_list, MAX_NUMBER, save_arch_str2op_list, padding_zeros, operation2integers, \
    delete_useless_node, API201
from Toy_experiment import model_random_forest_regressor, get_toy_data
import os
import pickle
import random
import copy
import numpy as np
from e2epp.e2epp import train_e2epp, test_e2epp
import argparse

expand = 1.1

NONE = 'none'
CONV1X1 = 'nor_conv_1x1'
CONV3X3 = 'nor_conv_3x3'
AP3X3 = 'avg_pool_3x3'
SKIP = 'skip_connect'


def op_list2str(op_list):
    op_str = '|{}~0|+|{}~0|{}~1|+|{}~0|{}~1|{}~2|'.format(op_list[0], op_list[1], op_list[2], op_list[3], op_list[4],
                                                          op_list[5])
    return op_str


def predict_by_predictor(predictor_name, population, predictor, integers2one_hot):
    population_metrics = get_metrics_from_population(population)
    X, _, __ = get_toy_data(population_metrics, create_more_metrics=False, select_upper_tri=False,
                            additional_metrics=False,
                            integers2one_hot=integers2one_hot)
    if predictor_name == 'HAAP':
        pred_y = predictor.predict(X)
    else:
        pred_y = test_e2epp(X, predictor[0], predictor[1])
    for index, acc in enumerate(pred_y):
        population[index]['acc'] = acc

    return population


def get_metrics_from_population(population):
    metrics = {}
    for index in population:
        individual = population[index]
        op_list = individual['arch']
        pruned_matrix, pruned_op = delete_useless_node(op_list)
        if pruned_matrix is None:
            individual['acc'] = -1
            continue

        padding_matrix, padding_op = padding_zeros(pruned_matrix, pruned_op)
        op_integers = operation2integers(padding_op)
        metrics[index] = {'final_training_time': -1, 'final_test_accuracy': -1}
        metrics[index]['fixed_metrics'] = {'module_adjacency': padding_matrix, 'module_integers': op_integers,
                                           'trainable_parameters': -1}
    return metrics


def population_initialization(ordered_dic, population_size=100):
    population = {}
    sample_list = list(range(0, MAX_NUMBER))
    expand_population_size = int(population_size * expand)
    index_list = random.sample(sample_list, expand_population_size)
    times = 0
    for index in index_list:
        if times == population_size:
            break
        op_list = save_arch_str2op_list(ordered_dic[index]['arch_str'])
        pruned_matrix, pruned_op = delete_useless_node(op_list)
        if pruned_matrix is None:
            continue
        population[times] = {}
        population[times]['arch'] = op_list
        population[times]['acc'] = 0
        times += 1
    return population


def binary_selection(population):
    population_size = len(population)
    random_two_integer = random.sample(list(range(0, population_size - 1)), 2)
    acc1 = population[random_two_integer[0]]['acc']
    acc2 = population[random_two_integer[1]]['acc']
    if acc1 > acc2:
        return random_two_integer[0]
    else:
        return random_two_integer[1]


def generate_offspring(population):
    num_repeate = len(population) // 2
    offspring = {}
    for i in range(num_repeate):
        arch_index1 = binary_selection(population)
        arch_index2 = copy.deepcopy(arch_index1)
        num_while = 0
        while (arch_index1 == arch_index2):
            arch_index2 = binary_selection(population)
            num_while += 1
            if num_while > 10:
                print('Choosing two architecture index too many times! Some Errors occur!')

        cross_prob = 0.9
        if random.random() < cross_prob:
            offspring1, offspring2 = crossover_operator(population, arch_index1, arch_index2)
            # check invalid individual
            pruned_matrix, pruned_op = delete_useless_node(offspring1)
            if pruned_matrix is None:
                offspring1 = population[arch_index1]['arch']

            pruned_matrix, pruned_op = delete_useless_node(offspring2)
            if pruned_matrix is None:
                offspring2 = population[arch_index2]['arch']

        else:
            offspring1, offspring2 = population[arch_index1]['arch'], population[arch_index2]['arch']

        mutation_prob = 0.2
        if random.random() < mutation_prob:
            mutation_offspring1 = mutation_operator(offspring1)
            mutation_offspring2 = mutation_operator(offspring2)
            # check invalid individual
            pruned_matrix, pruned_op = delete_useless_node(mutation_offspring1)
            if pruned_matrix is None:
                mutation_offspring1 = offspring1

            pruned_matrix, pruned_op = delete_useless_node(mutation_offspring2)
            if pruned_matrix is None:
                mutation_offspring2 = offspring2
        else:
            mutation_offspring1 = offspring1
            mutation_offspring2 = offspring2

        offspring[i * 2] = {'arch': mutation_offspring1, 'acc': 0}
        offspring[i * 2 + 1] = {'arch': mutation_offspring2, 'acc': 0}

    return offspring


def mutation_operator(offspring):
    # determine the mutation position
    mutation_position = random.randint(0, 5)

    mutation_offspring = copy.deepcopy(offspring)

    op_list = [NONE, CONV1X1, CONV3X3, AP3X3, SKIP]
    original_operation = mutation_offspring[mutation_position]
    op_list.remove(original_operation)
    force_mutation_op = random.choice(op_list)
    mutation_offspring[mutation_position] = force_mutation_op
    return mutation_offspring


def crossover_operator(population, index_arch1, index_arch2):
    # determine the crossover position
    crossover_position = random.randint(0, 5)
    new_arch1, new_arch2 = population[index_arch1]['arch'], population[index_arch2]['arch']
    temp_op = new_arch1[crossover_position]
    new_arch1[crossover_position] = new_arch2[crossover_position]
    new_arch2[crossover_position] = temp_op

    return new_arch1, new_arch2


def environment_selection(population, offspring):
    elitism_rate = 0.2
    next_population = {}
    population_size = len(population)
    elitism_num = int(population_size * elitism_rate)
    for index in offspring:
        population[population_size + index] = offspring[index]

    population_order = sorted(population.items(), key=lambda x: x[1]['acc'], reverse=True)
    for index in range(elitism_num):
        next_population[index] = {'arch': population_order[index][1]['arch'], 'acc': -1}
    for index in range(elitism_num):
        population_order.pop(0)
    for i in range(elitism_num, population_size):
        indi1, indi2 = random.sample(population_order, 2)
        if indi1[1]['acc'] > indi2[1]['acc']:
            better_indi = indi1
        else:
            better_indi = indi2
        next_population[i] = {'arch': better_indi[1]['arch'], 'acc': -1}
        population_order.remove(better_indi)

    return next_population


def GAon201(predictor_name, train_num, num_generation, create_more_metrics, integers2one_hot):
    # load data
    tidy_file = r'path/tidy_nas_bench_201.pkl'
    if not os.path.exists(tidy_file):
        raise Exception("Please run demo3.py first!")
    else:
        with open(tidy_file, 'rb') as file:
            ordered_dic = pickle.load(file)

    # start to sample train dataset
    expand_train_num = int(train_num * expand)
    sample_list = list(range(0, MAX_NUMBER))
    train_list = random.sample(sample_list, expand_train_num)
    train_list.sort()

    train_metrics = get_metrics_from_index_list(train_list, ordered_dic, train_num, 'cifar10_valid')
    X, y, _ = get_toy_data(train_metrics, create_more_metrics=create_more_metrics, select_upper_tri=False,
                           additional_metrics=False,
                           integers2one_hot=integers2one_hot)

    # initialize predictor and fit
    if predictor_name == 'HAAP':
        predictor = model_random_forest_regressor
        predictor.fit(X, y)
    else:
        e2epp_tree, e2epp_features = train_e2epp(X, y)
        predictor = e2epp_tree, e2epp_features

    # initialize population
    population = population_initialization(ordered_dic)

    for gen_no in range(num_generation):
        population = predict_by_predictor(predictor_name, population, predictor, integers2one_hot)
        print('start generate offspring {}'.format(gen_no))
        offspring = generate_offspring(population)
        offspring = predict_by_predictor(predictor_name, offspring, predictor, integers2one_hot)
        population = environment_selection(population, offspring)

    population = predict_by_predictor(predictor_name, population, predictor, integers2one_hot)
    return population


def query_by_arch(best_arch):
    tidy_file = r'path/tidy_nas_bench_201.pkl'
    if not os.path.exists(tidy_file):
        raise Exception("Please run demo3.py first!")
    else:
        with open(tidy_file, 'rb') as file:
            ordered_dic = pickle.load(file)
    for index in ordered_dic:
        arch = ordered_dic[index]['arch_str']
        if arch == best_arch:
            best_metric = ordered_dic[index]
            break

    return best_metric['cifar10'], best_metric['cifar10_valid200']


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Offline predictor')
    parser.add_argument('--predictor', type=str, choices=['HAAP', 'e2epp'], default='e2epp', help='name of predictor')
    parser.add_argument('--create-more-metrics', choices=[True, False], default=True, help='data augmentation')
    parser.add_argument('--integers2one-hot', choices=[True, False], default=True, help='one-hot encoding')
    args = parser.parse_args()

    create_more_metrics = args.create_more_metrics
    integers2one_hot = args.integers2one_hot
    num_generation = 20
    # the num_query is not the limit of the number of query architectures, but the time=4000 is the limitation
    num_query = 150

    repeat_num = 10
    best_acc_list = []
    best_valid_acc_list = []

    for _ in range(repeat_num):
        last_population = GAon201(args.predictor, num_query, num_generation, create_more_metrics, integers2one_hot)
        population_order = sorted(last_population.items(), key=lambda x: x[1]['acc'], reverse=True)
        best_arch = population_order[0][1]['arch']
        best_arch_str = op_list2str(best_arch)

        # query test acc
        # nasbench201 = API201(r'path/NAS-Bench-201-v1_0-e61699.pth')
        # best_acc = nasbench201.query_by_arch(best_arch_str)
        best_acc, best_valid_acc = query_by_arch(best_arch_str)
        print('Best acc: {}'.format(best_acc))
        best_acc_list.append(best_acc)
        best_valid_acc_list.append(best_valid_acc)

    print('Best acc mean: {}, std: {}'.format(np.mean(best_acc_list), np.std(best_acc_list)))
    print('Best valid_acc mean: {}, std: {}'.format(np.mean(best_valid_acc_list), np.std(best_valid_acc_list)))