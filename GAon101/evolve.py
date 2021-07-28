import numpy as np
import random
from nasbench import api
from GAon101.individual import Individual
from GAon101.population import Population
import copy
from GAon101.utils import utl2matrix, matrix2utl, population_log, write_best_individual
import os
import pickle

NASBENCH_TFRECORD = r'..\path\nasbench_only108.tfrecord'
nasbench = api.NASBench(NASBENCH_TFRECORD)


def query_fitness_for_indi(query_indi: Individual):
    print("query fitness for individual...")
    model_spec = api.ModelSpec(matrix=query_indi.indi['matrix'], ops=query_indi.indi['op_list'])
    _, computed_metrics = nasbench.get_metrics_from_spec(model_spec)
    final_test_accuracy_list = []
    final_training_time_list = []
    for i in range(3):
        # the 108 means the train epochs, and I have only downloaded the metrics in Epoch 108 (full train)
        # the three iterations: three results of independent experiments recorded in the dataset
        final_test_accuracy_list.append(computed_metrics[108][i]['final_test_accuracy'])
        final_training_time_list.append(computed_metrics[108][i]['final_training_time'])
    # use the mean of three metrics
    final_test_accuracy = np.mean(final_test_accuracy_list)
    final_training_time = np.mean(final_training_time_list)
    query_indi.mean_acc = final_test_accuracy
    query_indi.mean_training_time = final_training_time


# this is for population
def query_fitness(gen_no, query_pop: Population):
    print("query fitness for population {}".format(gen_no))
    for j, indi in enumerate(query_pop.pops):
        model_spec = api.ModelSpec(matrix=indi.indi['matrix'], ops=indi.indi['op_list'])
        _, computed_metrics = nasbench.get_metrics_from_spec(model_spec)
        final_test_accuracy_list = []
        final_training_time_list = []
        for i in range(3):
            # the 108 means the train epochs, and I have only downloaded the metrics in Epoch 108 (full train)
            # the three iterations: three results of independent experiments recorded in the dataset
            final_test_accuracy_list.append(computed_metrics[108][i]['final_test_accuracy'])
            final_training_time_list.append(computed_metrics[108][i]['final_training_time'])
        # use the mean of three metrics
        final_test_accuracy = np.mean(final_test_accuracy_list)
        final_training_time = np.mean(final_training_time_list)
        query_pop.pops[j].mean_acc = final_test_accuracy
        query_pop.pops[j].mean_training_time = final_training_time
    population_log(gen_no, query_pop)


class Evolution():
    def __init__(self, m_prob=0.2, m_num_matrix=1, m_num_op_list=1, x_prob=0.9, population_size=100):
        self.m_prob = m_prob
        self.x_prob = x_prob
        self.m_num_matrix = m_num_matrix
        self.m_num_op_list = m_num_op_list
        self.population_size = population_size

    def initialize_popualtion(self, type='GP'):
        print("initializing population with number {}...".format(self.population_size))
        init_path = r'../pkl/{}_init_population_{}.pkl'.format(type, self.population_size)
        if os.path.exists(init_path):
            # load population
            print('loading population...')
            with open(init_path, 'rb') as file:
                self.pops = pickle.load(file)
        else:
            self.pops = Population(self.population_size, self.m_num_matrix, self.m_num_op_list)
            with open(init_path, 'wb') as file:
                pickle.dump(self.pops, file)
        # all the initialized population should be saved
        population_log(0, self.pops)

    def recombinate(self, pop_size) -> Population:
        print("mutation and crossover...")
        offspring_list = []
        for _ in range(int(pop_size / 2)):
            p1 = self.tournament_selection()
            p2 = self.tournament_selection()
            # crossover
            if random.random() < self.x_prob:
                offset1, offset2 = self.crossover(p1, p2)
            else:
                offset1 = copy.deepcopy(p1)
                offset2 = copy.deepcopy(p2)
            # mutation
            if random.random() < self.m_prob:
                offset1.mutation()
            if random.random() < self.m_prob:
                offset2.mutation()
            offspring_list.append(offset1)
            offspring_list.append(offset2)
        offspring_pops = Population(0)
        offspring_pops.set_populations(offspring_list)
        return offspring_pops

    def environmental_selection(self, gen_no, offspring_population: Population):
        # environment selection from the current population and the offspring population
        assert (self.pops.get_pop_size() == self.population_size)
        assert (offspring_population.get_pop_size() == self.population_size)
        print('environmental selection...')
        elitism = 0.2
        e_count = int(self.population_size * elitism)
        indi_list = self.pops.pops
        indi_list.extend(offspring_population.pops)
        indi_list.sort(key=lambda x: x.mean_acc, reverse=True)
        # descending order
        elistm_list = indi_list[0:e_count]

        left_list = indi_list[e_count:]
        np.random.shuffle(left_list)

        for _ in range(self.population_size - e_count):
            i1 = random.randint(0, len(left_list) - 1)
            i2 = random.randint(0, len(left_list) - 1)
            winner = self.selection(left_list[i1], left_list[i2])
            elistm_list.append(winner)

        self.pops.set_populations(elistm_list)
        # # create the save_pops, because the self.pops may contains some individuals that don't have the real acc,
        # # actually, they contains the predicted acc
        # save_pops = copy.deepcopy(self.pops)
        # query_fitness(gen_no, save_pops)
        # population_log(gen_no, save_pops)
        population_log(gen_no, self.pops)
        write_best_individual(gen_no, self.pops)


    def crossover(self, p1: Individual, p2: Individual, utl_len=21, op_list_len=7):
        p1 = copy.deepcopy(p1)
        p2 = copy.deepcopy(p2)
        p1.clear_state_info()
        p2.clear_state_info()
        utl1 = matrix2utl(p1.indi['matrix'])
        utl2 = matrix2utl(p2.indi['matrix'])
        retry_num = 0
        while True:
            retry_num += 1
            matrix_cross_point = random.randint(1, utl_len - 1)
            crossed_utl1 = np.hstack((utl1[:matrix_cross_point], utl2[matrix_cross_point:]))
            crossed_utl2 = np.hstack((utl2[:matrix_cross_point], utl1[matrix_cross_point:]))
            crossed_matrix1 = utl2matrix(crossed_utl1)
            crossed_matrix2 = utl2matrix(crossed_utl2)
            model_spec1 = api.ModelSpec(matrix=crossed_matrix1, ops=p1.indi['op_list'])
            model_spec2 = api.ModelSpec(matrix=crossed_matrix2, ops=p2.indi['op_list'])
            # considering the invalid spec
            if model_spec1.valid_spec and (np.sum(model_spec1.matrix) <= 9) and model_spec2.valid_spec and (
                    np.sum(model_spec2.matrix) <= 9):
                break
            if retry_num > 20:
                print('Crossover has tried for more than 20 times, but still get invalid spec.\n'
                      'Give up this crossover and go on...')
                crossed_matrix1 = p1.indi['matrix']
                crossed_matrix2 = p2.indi['matrix']
                break
        p1.indi['matrix'] = crossed_matrix1
        p2.indi['matrix'] = crossed_matrix2

        op_list1 = p1.indi['op_list']
        op_list2 = p2.indi['op_list']
        op_list_cross_point = random.randint(1, op_list_len - 1)
        crossed_op_list1 = np.hstack((op_list1[:op_list_cross_point], op_list2[op_list_cross_point:]))
        crossed_op_list2 = np.hstack((op_list2[:op_list_cross_point], op_list2[op_list_cross_point:]))
        p1.indi['op_list'] = crossed_op_list1.tolist()
        p2.indi['op_list'] = crossed_op_list2.tolist()

        return p1, p2

    def tournament_selection(self):
        ind1_id = random.randint(0, self.pops.get_pop_size() - 1)
        ind2_id = random.randint(0, self.pops.get_pop_size() - 1)
        ind1 = self.pops.get_individual_at(ind1_id)
        ind2 = self.pops.get_individual_at(ind2_id)
        winner = self.selection(ind1, ind2)
        return winner

    def selection(self, ind1, ind2):
        if ind1.mean_acc > ind2.mean_acc:
            return ind1
        else:
            return ind2


if __name__ == '__main__':
    m_prob = 0.2
    m_num_matrix = 1
    m_num_op_list = 1
    x_prob = 0.9
    population_size = 100
    num_generation = 20

    total_training_time = 0

    Evolution = Evolution(m_prob, m_num_matrix, m_num_op_list, x_prob, population_size)
    Evolution.initialize_popualtion()
    gen_no = 0
    query_fitness(gen_no, Evolution.pops)
    total_training_time += Evolution.pops.calculate_population_training_time()
    while True:
        gen_no += 1
        if gen_no > num_generation:
            break
        offsprings = Evolution.recombinate(population_size)
        query_fitness(gen_no, offsprings)
        total_training_time += offsprings.calculate_population_training_time()
        Evolution.environmental_selection(gen_no, offsprings)

    save_path = r'pops_log\total_training_time.txt'
    with open(save_path, 'w') as file:
        file.write('Total_training_time: ' + str(total_training_time) + '\n')
        file.write('Total_training_num: ' + str(population_size * num_generation) + '\n')
        file.write(
            'm_prob: {}, m_num_matrix: {}, m_num_op_list: {}, x_prob: {}'.format(m_prob, m_num_matrix,
                                                                                 m_num_op_list,
                                                                                 x_prob))
