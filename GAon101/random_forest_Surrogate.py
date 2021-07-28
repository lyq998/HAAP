from GAon101.evolve import Evolution, query_fitness, query_fitness_for_indi
from sklearn.ensemble import RandomForestRegressor
from GAon101.population import Population
import numpy as np
import copy
from get_data_from_101 import delete_margin
from GAon101.utils import operations2onehot, GP_log, population_log, NULL
from nasbench import api
from GAon101.individual import Individual
from Create_more_metrics import create_new_metrics


# transform genotype population to phenotype, and pad the matrix and the op_list to max length
def genotype2phenotype(pops: Population) -> Population:
    genotype_population = pops
    phenotype_population = Population(0)
    for indi in genotype_population.pops:
        matrix = indi.indi['matrix']
        op_list = indi.indi['op_list']
        model_spec = api.ModelSpec(matrix, op_list)
        pruned_matrix = model_spec.matrix
        pruned_op_list = model_spec.ops
        # start to padding zeros to 7*7 matrix and 7 op_list
        len_operations = len(pruned_op_list)
        assert len_operations == len(pruned_matrix)
        padding_matrix = copy.deepcopy(pruned_matrix)
        if len_operations != 7:
            for j in range(len_operations, 7):
                pruned_op_list.insert(j - 1, NULL)

            padding_matrix = np.insert(pruned_matrix, len_operations - 1,
                                       np.zeros([7 - len_operations, len_operations]), axis=0)
            padding_matrix = np.insert(padding_matrix, [len_operations - 1], np.zeros([7, 7 - len_operations]), axis=1)
        phenotype_individual = Individual()
        phenotype_individual.create_an_individual(padding_matrix, pruned_op_list)
        phenotype_individual.mean_acc = indi.mean_acc
        phenotype_individual.mean_training_time = indi.mean_training_time
        phenotype_population.add_individual(phenotype_individual)
    return phenotype_population


def get_input_X(pops: Population):
    X = []
    for indi in pops.pops:
        input_metrix = []
        matrix = indi.indi['matrix']
        matrix = delete_margin(np.array(matrix))
        flattend_matrix = np.reshape(matrix, (-1)).tolist()
        input_metrix.extend(flattend_matrix)

        op_list = indi.indi['op_list']
        op_list = operations2onehot(op_list[1: -1])
        input_metrix.extend(op_list)

        X.append(input_metrix)
    return X


# Matrix and Op_list to vector
def MO2vector(matrix, op_list):
    input_metrix = []

    matrix = delete_margin(matrix)
    flattend_matrix = np.reshape(matrix, (-1)).tolist()
    input_metrix.extend(flattend_matrix)

    op_list = operations2onehot(op_list)
    input_metrix.extend(op_list)

    return input_metrix


# this is for genotype surrogate evolution
class RF_evolution(Evolution):
    def __init__(self, num_estimators, m_prob=0.2, m_num_matrix=1, m_num_op_list=1, x_prob=0.9,
                 population_size=100):
        super(RF_evolution, self).__init__(m_prob, m_num_matrix, m_num_op_list, x_prob, population_size)
        self.num_estimators = num_estimators
        self.archive = Population(0)

    def init_RF_model(self):
        self.RFmodel = RandomForestRegressor(n_estimators=self.num_estimators)

    def fit_RF_model(self, data_augmentation=True):
        X = []
        y = []
        for indi in self.archive.pops:
            matrix = indi.indi['matrix']
            op_list = indi.indi['op_list']
            if data_augmentation:
                more_metrics = create_new_metrics(matrix, op_list[1: -1], select_upper_tri=False, max_num=120)
                for same_metric in more_metrics:
                    adjacent_matrix, module_integers = same_metric['module_adjacency'], same_metric['module_integers']
                    input_metrix = MO2vector(adjacent_matrix, module_integers)
                    X.append(input_metrix)
                    y.append(indi.mean_acc)
            else:
                input_metrix = MO2vector(matrix, op_list[1: -1])
                X.append(input_metrix)
                y.append(indi.mean_acc)
        self.RFmodel.fit(X, y)

    def predict_by_RF(self, pred_pops: Population, phenotype=False):
        if phenotype:
            pred_pops_new = genotype2phenotype(pred_pops)
            X = get_input_X(pred_pops_new)
        else:
            X = get_input_X(pred_pops)
        pred_y = self.RFmodel.predict(X)
        for i, indi in enumerate(pred_pops.pops):
            pred_pops.pops[i].mean_acc = pred_y[i]
            # using the GP model, don't need to training
            pred_pops.pops[i].mean_training_time = 0


if __name__ == '__main__':
    m_prob = 0.2
    m_num_matrix = 1
    m_num_op_list = 1
    x_prob = 0.9
    population_size = 100
    num_generation = 20
    num_resample = 0
    num_estimators = 230
    archive_num = 1000
    # phenotype=True means using phenotype to predict and fitting the GPmodel with phenotype
    # phenotype=False means using genotype to predict and fitting the GPmodel with genotype
    phenotype = True
    surrogate = True

    total_training_time = 0
    final_one_acc = []

    RF_Evolution = RF_evolution(num_estimators, m_prob, m_num_matrix, m_num_op_list, x_prob, population_size)
    RF_Evolution.initialize_popualtion(type='RF')
    RF_Evolution.init_RF_model()
    gen_no = 0
    # query_fitness(gen_no, RF_Evolution.pops)
    # total_training_time += RF_Evolution.pops.calculate_population_training_time()
    if surrogate:
        fit_archive = Population(archive_num)
        query_fitness(-1, fit_archive)
        if not phenotype:
            RF_Evolution.archive = copy.deepcopy(fit_archive)
        else:  # if phenotype:
            RF_Evolution.archive = genotype2phenotype(fit_archive)
            RF_Evolution.fit_RF_model()

    if not surrogate:
        query_fitness(gen_no, RF_Evolution.pops)
    else:
        RF_Evolution.predict_by_RF(RF_Evolution.pops, phenotype=phenotype)
    while True:
        gen_no += 1
        if gen_no > num_generation:
            break
        offspring = RF_Evolution.recombinate(population_size)
        if not surrogate:
            query_fitness(gen_no, offspring)
        else:
            RF_Evolution.predict_by_RF(offspring, phenotype=phenotype)

        RF_Evolution.environmental_selection(gen_no, offspring)

    # for the last generation
    last_resample_num = 1
    sorted_acc_index = RF_Evolution.pops.get_sorted_index_order_by_acc()
    last_population = Population(0)
    for i in sorted_acc_index[:last_resample_num]:
        last_population.add_individual(RF_Evolution.pops.get_individual_at(i))

    gen_no = 'final'
    query_fitness(gen_no, last_population)
    population_log(gen_no, last_population)
    final_one_acc.append(last_population.pops[0].mean_acc)

    save_path = r'pops_log\total_training_time.txt'
    with open(save_path, 'w') as file:
        file.write('Total_training_time: ' + str(total_training_time) + '\n')
        file.write('Total_training_num: ' + str(population_size + num_generation * num_resample) + '\n')
        file.write(
            'm_prob: {}, m_num_matrix: {}, m_num_op_list: {}, x_prob: {}\n'.format(m_prob, m_num_matrix, m_num_op_list,
                                                                                   x_prob))
        file.write('RF_surrogate: True, num_resample: {}, phenotype: {}'.format(num_resample, phenotype))

    print('The ACC of the best individual found is: {}'.format(final_one_acc))
