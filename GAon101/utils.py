import numpy as np

INPUT = 'input'
OUTPUT = 'output'
CONV1X1 = 'conv1x1-bn-relu'
CONV3X3 = 'conv3x3-bn-relu'
MAXPOOL3X3 = 'maxpool3x3'
NULL = 'null'


# matrix to upper triangular list
def matrix2utl(matrix):
    utl = []
    assert len(matrix) == len(matrix[0])
    matrix_len = len(matrix)
    for i in range(matrix_len - 1):
        utl += list(matrix[i][i + 1:])
    # flatten to 1-d
    utl = np.reshape(utl, -1)
    return utl


# upper triangular list to matrix
def utl2matrix(utl, matrix_len=7):
    matrix = np.zeros((matrix_len, matrix_len), dtype='int8')
    start_index = 0
    for i in range(matrix_len - 1):
        cur_len = matrix_len - i - 1
        matrix[i][i + 1:] = utl[start_index:start_index + cur_len]
        start_index += cur_len

    return matrix


# transform operation list to one-hot list
# input:: op_list:str type
# output:: one-hot_op_list:one-hot
def operations2onehot(op_list):
    dict_oper2one_hot = {NULL: [0, 0, 0, 1], CONV1X1: [0, 0, 1, 0], CONV3X3: [0, 1, 0, 0], MAXPOOL3X3: [1, 0, 0, 0]}
    module_one_hot = np.array([dict_oper2one_hot[x] for x in op_list])
    # use [1: -1] to remove 'input' and 'output'
    module_one_hot = np.reshape(module_one_hot, (-1))
    module_one_hot = module_one_hot.tolist()
    return module_one_hot


def population_log(gen_no, pops):
    save_path = r'pops_log\gen_{}.txt'.format(gen_no)
    with open(save_path, 'w') as myfile:
        myfile.write(str(pops))
        myfile.write("\n")


def write_best_individual(gen_no, pops):
    arg_index = pops.get_sorted_index_order_by_acc()
    best_individual = pops.get_individual_at(arg_index[0])
    save_path = r'pops_log\best_acc.txt'
    with open(save_path, 'a') as myfile:
        myfile.write('gen_no: {}'.format(gen_no) + '\n')
        myfile.write(str(best_individual))
        myfile.write('\n')


def GP_log(gen_no, query_pops, left_offspring):
    save_path = r'pops_log\GP_{}.txt'.format(gen_no)
    with open(save_path, 'w') as myfile:
        myfile.write('query_pops\n')
        myfile.write(str(query_pops))
        myfile.write("\n")
        myfile.write('left_offspring\n')
        myfile.write(str(left_offspring))


if __name__ == '__main__':
    # matrix = [[0, 1, 1, 1, 0, 1, 0],  # input layer
    #           [0, 0, 0, 0, 0, 0, 0],  # 1x1 conv
    #           [0, 0, 0, 1, 0, 0, 0],  # 3x3 conv
    #           [0, 0, 0, 0, 1, 0, 0],  # 5x5 conv (replaced by two 3x3's)
    #           [0, 0, 0, 0, 0, 0, 0],  # 5x5 conv (replaced by two 3x3's)
    #           [0, 0, 0, 0, 0, 0, 0],  # 3x3 max-pool
    #           [0, 0, 0, 0, 0, 0, 0]]
    # print(matrix)
    # utl = matrix2utl(matrix)
    # print(utl)
    # matrix = utl2matrix(utl)
    # print(matrix)
    op_list = [INPUT, CONV1X1, CONV3X3, CONV3X3, CONV3X3, MAXPOOL3X3, OUTPUT]
    op_onehot = operations2onehot(op_list)
    print()
