import h5py
import numpy as np
from torch.utils.data import Dataset
from architecture_augmentation import create_new_metrics


class Nb101DatasetAug(Dataset):
    MEAN = 0.908192
    STD = 0.023961

    def __init__(self, split=None):
        self.num_vertices = []
        self.adjacency = []
        self.operations = []
        self.valid_accuracy = []
        self.org_id = []
        self.aug_id = []
        temp_aug_id = 0
        with h5py.File("data/nasbench.hdf5", mode="r") as f:
            num_vertices = f["num_vertices"][()]
            adjacency = f["adjacency"][()]
            operations = f["operations"][()]
            metrics = f["metrics"][()]
        self.random_state = np.random.RandomState(0)
        if split is not None and split != "all":
            sample_range = np.load("data/train.npz")[str(split)]
        else:
            raise KeyError('please input the right split.')

        for org_index, index in enumerate(sample_range):
            # architecture augmentation
            org_num_vertices = num_vertices[index]
            org_adjacency = adjacency[index]
            org_operations = operations[index]
            org_metrics = metrics[index]

            if org_num_vertices != 7:
                # padding, exchange the input of the output node to the last column of the adjacency and exchange the
                # output node to the last item of org_operations
                org_adjacency[:, [org_num_vertices - 1, 6]] = org_adjacency[:, [6, org_num_vertices - 1]]
                # -3 is none type
                org_operations[org_num_vertices:] = -3
                org_operations[[org_num_vertices - 1, 6]] = org_operations[[6, org_num_vertices - 1]]

            aug_arch = self.architecture_augmentation(org_num_vertices, org_adjacency, org_operations, org_metrics)

            for arch in aug_arch:
                self.num_vertices.append(arch['num_vertices'])
                self.adjacency.append(arch['module_adjacency'])
                self.operations.append(arch['module_integers'])
                self.valid_accuracy.append(arch['valid_accuracy'])
                self.org_id.append(org_index)
                self.aug_id.append(temp_aug_id)
                temp_aug_id += 1

    def architecture_augmentation(self, org_num_vertices, org_adjacency, org_operations, org_metrics):
        valid_accuracy = org_metrics[-1, 0, -1, 2]
        architecture_augmentation = create_new_metrics(org_adjacency, org_operations[1:-1])
        # adjust type
        for index, _ in enumerate(architecture_augmentation):
            tmp_list = [-1] + list(architecture_augmentation[index]['module_integers']) + [-2]
            architecture_augmentation[index]['module_integers'] = np.array(tmp_list)
            architecture_augmentation[index]['num_vertices'] = org_num_vertices
            architecture_augmentation[index]['valid_accuracy'] = valid_accuracy

        return architecture_augmentation

    def __len__(self):
        return len(self.adjacency)

    @classmethod
    def normalize(cls, num):
        return (num - cls.MEAN) / cls.STD

    @classmethod
    def denormalize(cls, num):
        return num * cls.STD + cls.MEAN

    def __getitem__(self, index):
        n = self.num_vertices[index]
        ops_onehot = np.array([[i == k + 2 for i in range(5)] for k in self.operations[index]], dtype=np.float32)

        val_acc = self.valid_accuracy[index]
        result = {
            "num_vertices": n,
            "adjacency": self.adjacency[index],
            "operations": ops_onehot,
            "mask": np.array([i < n for i in range(7)], dtype=np.float32),
            "val_acc": self.normalize(val_acc),
            "org_id": self.org_id[index],
            "aug_id": self.aug_id[index]
        }
        return result


class Nb101Dataset(Dataset):
    MEAN = 0.908192
    STD = 0.023961

    def __init__(self, split=None, debug=False, arch_aug=False):
        self.arch_aug = arch_aug
        self.hash2id = dict()
        with h5py.File("data/nasbench.hdf5", mode="r") as f:
            for i, h in enumerate(f["hash"][()]):
                self.hash2id[h.decode()] = i
            self.num_vertices = f["num_vertices"][()]
            self.trainable_parameters = f["trainable_parameters"][()]
            self.adjacency = f["adjacency"][()]
            self.operations = f["operations"][()]
            self.metrics = f["metrics"][()]
        self.random_state = np.random.RandomState(0)
        if split is not None and split != "all":
            self.sample_range = np.load("data/train.npz")[str(split)]
        else:
            self.sample_range = list(range(len(self.hash2id)))
        self.debug = debug
        self.seed = 0

    def __len__(self):
        return len(self.sample_range)

    def _check(self, item):
        n = item["num_vertices"]
        ops = item["operations"]
        adjacency = item["adjacency"]
        mask = item["mask"]
        assert np.sum(adjacency) - np.sum(adjacency[:n, :n]) == 0
        assert np.sum(ops) == n
        assert np.sum(ops) - np.sum(ops[:n]) == 0
        assert np.sum(mask) == n and np.sum(mask) - np.sum(mask[:n]) == 0

    def mean_acc(self):
        return np.mean(self.metrics[:, -1, self.seed, -1, 2])

    def std_acc(self):
        return np.std(self.metrics[:, -1, self.seed, -1, 2])

    @classmethod
    def normalize(cls, num):
        return (num - cls.MEAN) / cls.STD

    @classmethod
    def denormalize(cls, num):
        return num * cls.STD + cls.MEAN

    def resample_acc(self, index, split="val"):
        # when val_acc or test_acc are out of range
        assert split in ["val", "test"]
        split = 2 if split == "val" else 3
        for seed in range(3):
            acc = self.metrics[index, -1, seed, -1, split]
            if not self._is_acc_blow(acc):
                return acc
        if self.debug:
            print(index, self.metrics[index, -1, :, -1])
            raise ValueError
        return np.array(self.MEAN)

    def _is_acc_blow(self, acc):
        return acc < 0.2

    def __getitem__(self, index):
        index = self.sample_range[index]
        val_acc, test_acc = self.metrics[index, -1, self.seed, -1, 2:]
        if self._is_acc_blow(val_acc):
            val_acc = self.resample_acc(index, "val")
        if self._is_acc_blow(test_acc):
            test_acc = self.resample_acc(index, "test")
        n = self.num_vertices[index]
        if n < 7 and self.arch_aug:
            self.adjacency[index][:, [n - 1, 6]] = self.adjacency[index][:, [6, n - 1]]
            # -3 is none type
            self.operations[index][n:] = -3
            self.operations[index][[n - 1, 6]] = self.operations[index][[6, n - 1]]
        ops_onehot = np.array([[i == k + 2 for i in range(5)] for k in self.operations[index]], dtype=np.float32)
        if n < 7 and not self.arch_aug:
            ops_onehot[n:] = 0.
        result = {
            "num_vertices": n,
            "adjacency": self.adjacency[index],
            "operations": ops_onehot,
            "mask": np.array([i < n for i in range(7)], dtype=np.float32),
            "val_acc": self.normalize(val_acc),
            "test_acc": self.normalize(test_acc)
        }
        if self.debug:
            self._check(result)
        return result


if __name__ == '__main__':
    dataset = Nb101DatasetAug("334")
    from torch.utils.data import DataLoader

    data_loader = DataLoader(dataset, batch_size=10, shuffle=True)
    for step, batch in enumerate(data_loader):
        print('')
