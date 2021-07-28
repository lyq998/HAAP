from GAon101.individual import Individual
import numpy as np


class Population:
    def __init__(self, num_pops, m_num_matrix=1, m_num_op_list=1):
        self.num_pops = num_pops
        self.pops = []
        for i in range(num_pops):
            indi = Individual(m_num_matrix, m_num_op_list)
            indi.initialize()
            self.pops.append(indi)

    def copy_acc_time_from_Population(self, copy_population):
        assert len(self.pops) == len(copy_population.pops)
        for i, indi in enumerate(copy_population.pops):
            self.pops[i].mean_acc = indi.mean_acc
            self.pops[i].mean_training_time = indi.mean_training_time

    def calculate_population_training_time(self):
        total_training_time = 0
        for i in range(self.get_pop_size()):
            indi_i = self.get_individual_at(i)
            total_training_time += indi_i.mean_training_time

        return total_training_time

    def get_individual_at(self, i):
        return self.pops[i]

    def get_pop_size(self):
        return len(self.pops)

    def set_populations(self, new_pops):
        self.pops = new_pops

    # append new pops to the current self.pops
    def merge_populations(self, new_pops):
        for indi in new_pops:
            self.pops.append(indi)

    def add_individual(self, new_indi: Individual):
        self.pops.append(new_indi)

    def get_best_acc(self):
        mean_acc_list = []
        for i in range(self.get_pop_size()):
            indi = self.get_individual_at(i)
            mean_acc_list.append(indi.mean_acc)
        return np.max(mean_acc_list)

    def get_sorted_index_order_by_acc(self):
        mean_acc_list = []
        for i in range(self.get_pop_size()):
            indi = self.get_individual_at(i)
            mean_acc_list.append(indi.mean_acc)
        arg_index = np.argsort(-1 * np.array(mean_acc_list))
        return arg_index

    def __str__(self):
        _str = []
        arg_index = self.get_sorted_index_order_by_acc()
        for i in arg_index:
            _str.append(str(self.get_individual_at(i)))
        return '\n'.join(_str)


if __name__ == '__main__':
    pop = Population(num_pops=20, m_num_matrix=2, m_num_op_list=2)
    print(pop)
