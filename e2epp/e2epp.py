import numpy as np
from sklearn.tree import DecisionTreeRegressor
import pandas as pd
import matplotlib.pyplot as plt


def load_data():
    d = pd.read_csv('data.txt').values
    data = d[:, 1:-1]
    label = d[:, -1]
    return data, label


def make_decision_trees(train_data, train_label, n_tree):
    feature_record = []
    tree_record = []
    train_data = np.array(train_data)
    train_label = np.array(train_label)
    for _ in range(n_tree):
        sample_idx = np.arange(train_data.shape[0])
        np.random.shuffle(sample_idx)
        train_data = train_data[sample_idx, :]
        train_label = train_label[sample_idx]

        feature_idx = np.arange(train_data.shape[1])
        np.random.shuffle(feature_idx)
        n_feature = np.random.randint(1, train_data.shape[1] + 1)
        selected_feature_ids = feature_idx[0:n_feature]
        feature_record.append(selected_feature_ids)

        dt = DecisionTreeRegressor()
        dt.fit(train_data[:, selected_feature_ids], train_label)
        tree_record.append(dt)
    return tree_record, feature_record


def predict(test_data, trees, feature_ids):
    predict_list = []
    for tree, feature in zip(trees, feature_ids):
        predict_y = tree.predict(test_data[:, feature])
        predict_list.append(predict_y)
    return predict_list


def train_e2epp(train_data, train_label):
    n_tree = 1000
    trees, features = make_decision_trees(train_data, train_label, n_tree)
    return trees, features


def test_e2epp(test_data, trees, features):
    test_data = np.array(test_data)
    total_predict_list = np.zeros((len(trees), test_data.shape[0]))
    for i, (tree, feature) in enumerate(zip(trees, features)):
        predict_list = tree.predict(test_data[:, feature])
        total_predict_list[i, :] = predict_list
    predict_mean_y = np.mean(total_predict_list, 0)
    return predict_mean_y


def one_run(train_data, train_label, test_data, test_label):
    n_tree = 1000
    trees, features = make_decision_trees(train_data, train_label, n_tree)
    test_data = np.array(test_data)
    test_label = np.array(test_label)
    total_predict_list = np.zeros((len(trees), test_label.shape[0]))
    for i, (tree, feature) in enumerate(zip(trees, features)):
        predict_list = tree.predict(test_data[:, feature])
        total_predict_list[i, :] = predict_list
    predict_mean_y = np.mean(total_predict_list, 0)
    diff = np.mean(np.square(predict_mean_y - test_label))
    return diff, predict_mean_y, test_label


def one_run_for_each(train_data, train_label, test_data, test_label):
    n_tree = 1000
    trees, features = make_decision_trees(train_data, train_label, n_tree)
    test_num = test_data.shape[0]
    predict_labels = np.zeros(test_num)
    for i in range(test_num):
        this_test_data = test_data[i, :]
        predict_this_list = np.zeros(n_tree)

        for j, (tree, feature) in enumerate(zip(trees, features)):
            predict_this_list[j] = tree.predict([this_test_data[feature]])[0]

        # find the top 100 prediction
        predict_this_list = np.sort(predict_this_list)
        predict_this_list = predict_this_list[::-1]
        this_predict = np.mean(predict_this_list[0:100])
        predict_labels[i] = this_predict
    print(np.sqrt(np.mean(np.square(predict_labels - test_label))))
    print(np.mean(np.abs(predict_labels - test_label)))

    plt.plot(predict_labels, label='predict')
    plt.plot(test_label, label='true')
    plt.legend()
    plt.show()


def train_final_model():
    n_tree = 5000
    train_data, train_label = load_data()
    trees, features = make_decision_trees(train_data, train_label, n_tree)
    model = [trees, features]
    import joblib
    joblib.dump(model, 'predict_model.pkl')


def test_saved_model():
    n_tree = 5000
    import joblib
    test_data, test_label = load_data()
    test_num = test_data.shape[0]
    trees, features = joblib.load('predict_model.pkl')
    print(type(trees), len(trees))

    predict_labels = np.zeros(test_num)
    for i in range(test_num):
        this_test_data = test_data[i, :]
        predict_this_list = np.zeros(n_tree)

        for j, (tree, feature) in enumerate(zip(trees, features)):
            predict_this_list[j] = tree.predict([this_test_data[feature]])[0]

        # find the top 100 prediction
        predict_this_list = np.sort(predict_this_list)
        predict_this_list = predict_this_list[::-1]
        this_predict = np.mean(predict_this_list)
        predict_labels[i] = this_predict
    print(np.sqrt(np.mean(np.square(predict_labels - test_label))))
    print(np.mean(np.abs(predict_labels - test_label)))

    plt.plot(predict_labels, label='predict')
    plt.plot(test_label, label='true')
    plt.legend()
    plt.show()


def test_one_this_run():
    data, label = load_data()
    idx = np.arange(label.shape[0])
    #     np.random.shuffle(idx)
    #     data = data[idx,:]
    #     label = label[idx]
    train_num = int(idx.shape[0] * 0.8)
    train_data = data[0:train_num, :]
    train_label = label[0:train_num]
    test_data = data[train_num:, :]
    test_label = label[train_num:]
    err = one_run_for_each(train_data, train_label, test_data, test_label)
    print(err)


def ten_folds():
    folds = 10
    data, label = load_data()
    fold_size = 10  # data.shape[0]//folds
    fold_mean = np.zeros(folds)
    predict_y_list = []
    true_y_list = []
    for i in range(folds):
        start_index = i * fold_size
        end_index = (i + 1) * fold_size
        copy_data = np.copy(data)
        copy_label = np.copy(label)
        test_data = copy_data[start_index:end_index, :]
        test_label = copy_label[start_index:end_index]
        train_data = np.delete(copy_data, np.arange(start_index, end_index), axis=0)
        train_label = np.delete(copy_label, np.arange(start_index, end_index))
        diff, predict_y, true_y = one_run(train_data, train_label, test_data, test_label)
        fold_mean[i] = diff
        predict_y_list.append(predict_y)
        true_y_list.append(true_y)
    print('MSE:', np.mean(fold_mean))


if __name__ == '__main__':
    ten_folds()
