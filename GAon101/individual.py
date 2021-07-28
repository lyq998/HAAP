import numpy as np
import random
from nasbench import api
from GAon101.utils import utl2matrix, matrix2utl
import copy

INPUT = 'input'
OUTPUT = 'output'
CONV1X1 = 'conv1x1-bn-relu'
CONV3X3 = 'conv3x3-bn-relu'
MAXPOOL3X3 = 'maxpool3x3'
NULL = 'null'


class Individual:
    def __init__(self, m_num_matrix=1, m_num_op_list=1):
        self.indi = {}
        self.m_num_matrix = m_num_matrix
        self.m_num_op_list = m_num_op_list
        self.mean_acc = 0
        self.mean_training_time = 0

    def clear_state_info(self):
        self.mean_acc = 0
        self.mean_training_time = 0

    def create_an_individual(self, matrix, op_list):
        self.indi['matrix'] = matrix
        self.indi['op_list'] = op_list

    def initialize(self):
        self.indi['matrix'], self.indi['op_list'] = self.init_one_individual()

    def init_one_individual(self):
        # initial op_list
        op_list = []
        op_list.append(INPUT)
        for _ in range(5):
            rand_int = random.randint(1, 3)
            if rand_int == 1:
                op_list.append(CONV1X1)
            elif rand_int == 2:
                op_list.append(CONV3X3)
            else:  # rand_int==3
                op_list.append(MAXPOOL3X3)
        op_list.append(OUTPUT)

        matrix = np.zeros(shape=(7, 7), dtype='int8')
        model_spec = api.ModelSpec(matrix=matrix, ops=op_list)

        # if the matrix contains more than nine edges or the matrix is invalid (the graph is disconnected),
        # then reinitialize
        # Please note that, the model_spec.matrix is not the same with the matrix,
        # because the model_spec.matrix has pruned the matrix.
        while (not model_spec.valid_spec) or (np.sum(model_spec.matrix) > 9):
            # print('Start to initial a individual')
            matrix = np.zeros(shape=(7, 7), dtype='int8')
            # initial matrix by row, and the matrix must be upper triangular
            # the first row must contains at least one 1, and the last row must be all zeros
            # the middle five rows and have a 1/4 probability to be all zeros, and otherwise, it must contain
            # at least one 1
            # row 0 to 6
            for i in range(0, 6):
                if random.random() < 0.75 or i == 0:
                    num_ones = random.randint(1, 6 - i)
                    one_index = random.sample(range(1 + i, 7), num_ones)
                    for j in one_index:
                        matrix[i][j] = 1
                # else, this row are all zeros
            model_spec = api.ModelSpec(matrix=matrix, ops=op_list)

        return matrix, op_list

    def set_mean_acc(self, mean_acc):
        self.mean_acc = mean_acc

    def mutation(self):
        self.matrix_mutation()
        self.op_list_mutation()

    def matrix_mutation(self):
        def point_flip(point):
            if point == 0:
                return 1
            else:  # point==1
                return 0

        # avoid produce invalid matrix
        while True:
            # sample the flip points from 21 positions
            flip_positions = random.sample(range(21), self.m_num_matrix)
            utl = matrix2utl(self.indi['matrix'])
            for index in flip_positions:
                utl[index] = point_flip(utl[index])
            matrix = utl2matrix(utl)
            model_spec = api.ModelSpec(matrix=matrix, ops=self.indi['op_list'])
            if model_spec.valid_spec and (np.sum(model_spec.matrix) <= 9):
                break

        self.indi['matrix'] = utl2matrix(utl)

    def op_list_mutation(self):
        mutation_positions = random.sample(range(1, 6), self.m_num_op_list)
        op_list = copy.deepcopy(self.indi['op_list'])
        for index in mutation_positions:
            cur_op = op_list[index]
            if cur_op == CONV1X1:
                new_op = random.choice([CONV3X3, MAXPOOL3X3])
            elif cur_op == CONV3X3:
                new_op = random.choice([CONV1X1, MAXPOOL3X3])
            elif cur_op == MAXPOOL3X3:
                new_op = random.choice([CONV1X1, CONV3X3])
            # elif cur_op == NULL:
            #     new_op = random.choice([CONV1X1, CONV3X3, MAXPOOL3X3])
            else:
                raise ValueError(
                    'The op should be in [CONV1X1, CONV3X3, MAXPOOL3X3], but it is: {}'.format(cur_op))
            op_list[index] = new_op
        self.indi['op_list'] = op_list

    def __str__(self):
        str_ = []
        str_.append('Matrix:{}, Op_list:{}'.format(self.indi['matrix'], self.indi['op_list']))
        str_.append('Mean_ACC:{:.16f}'.format(self.mean_acc))
        str_.append('Mean_training_time:{}'.format(self.mean_training_time))

        return ', '.join(str_)


if __name__ == '__main__':
    indi = Individual()
    indi.initialize()
    print(indi)
    indi.mutation()
    print(indi)
