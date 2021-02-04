import os
import array
import time
import pickle
from pathlib import Path
from collections import OrderedDict
import json
import numpy as np


def pickle_loader(pkl_path):
    tic = time.time()
    # print("loading features from {}".format(pkl_path))
    with open(pkl_path, "rb") as f:
        data = pickle.load(f, encoding="latin1")
    # print("done in {:.3f}s".format(time.time() - tic))
    return data


def np_loader(np_path, l2norm=False):
    tic = time.time()
    print("loading features from {}".format(np_path))
    with open(np_path, "rb") as f:
        data = np.load(f, encoding="latin1", allow_pickle=True)
    print("done in {:.3f}s".format(time.time() - tic))
    if isinstance(data, np.ndarray) and data.size == 1:
        data = data[()]  # handle numpy dict storage convnetion
    if l2norm:
        print("L2 normalizing features")
        if isinstance(data, dict):
            for key in data:
                feats_ = data[key]
                feats_ = feats_ / max(np.linalg.norm(feats_), 1E-6)
                data[key] = feats_
        elif data.ndim == 2:
            data_norm = np.linalg.norm(data, axis=1)
            data = data / np.maximum(data_norm.reshape(-1, 1), 1E-6)
        else:
            raise ValueError("unexpected data format {}".format(type(data)))
    return data


def memcache(path):
    suffix = Path(path).suffix
    if suffix in {".pkl", ".pickle"}:
        res = pickle_loader(path)
    elif suffix == ".npy":
        res = np_loader(path)
    else:
        raise ValueError(f"unknown suffix: {suffix}")
    return res


def read_json(fname):
    with fname.open('rt') as handle:
        return json.load(handle, object_hook=OrderedDict)


def write_json(content, fname):
    with fname.open('wt') as handle:
        json.dump(content, handle, indent=4, sort_keys=False)


def compute_dims(config):
    ordered = sorted(config['experts']['modalities'])
    try:
        backbone = config["arch"]["args"]["backbone"]
    except KeyError:
        backbone = 'resnet'

    dims = []
    # Modality: im_feat, im_moe0, inter-1
    for expert in ordered:
        if expert == "im_feat" or "spatial" in expert or "im_moe" in expert:
            if backbone in ["resnet", "resnet152", "senet154", "polynet"]:
                out_dim = 2048
            elif backbone == "densenet":
                out_dim = 1664
            elif backbone == "inceptionresnetv2":
                out_dim = 1536
            elif backbone == "pnasnet5large":
                out_dim = 4320
            elif backbone == "nasnetalarge":
                out_dim = 4032
            elif expert == "vgg":
                out_dim = 4096
            else:
                raise ValueError
        elif expert == 'inter-1':
            if backbone == "densenet": # After transition3
                out_dim = 640
            if backbone == "resnet":
                out_dim = 1024
            if backbone == "resnet152":
                out_dim = 1024
        elif expert == 'inter-2':
            if backbone == "densenet": # After transition2
                out_dim = 256
            if backbone == "resnet":
                out_dim = 512
            if backbone == "resnet152":
                out_dim = 512
        else:
            out_dim = config["experts"]["ce_shared_dim"]
        dims.append((expert, out_dim))
    expert_dims = OrderedDict(dims)

    return expert_dims

class HashableOrderedDict(OrderedDict):
    def __hash__(self):
        return hash(tuple(sorted(self.items())))


class BigFile:

    def __init__(self, datadir):
        self.nr_of_images, self.ndims = map(int, open(os.path.join(datadir,'shape.txt')).readline().split())
        id_file = os.path.join(datadir, "id.txt")
        self.names = open(id_file, 'rb').read().strip().split()
        assert(len(self.names) == self.nr_of_images)
        self.name2index = dict(zip(self.names, range(self.nr_of_images)))
        self.binary_file = os.path.join(datadir, "feature.bin")
        print("[%s] %dx%d instances loaded from %s" % (self.__class__.__name__, self.nr_of_images, self.ndims, datadir))

    def read(self, requested, isname=True):
        requested = set(requested)
        if isname:
            index_name_array = [(self.name2index[x], x) for x in requested if x in self.name2index]
        else:
            assert(min(requested)>=0)
            assert(max(requested)<len(self.names))
            index_name_array = [(x, self.names[x]) for x in requested]
        if len(index_name_array) == 0:
            return [], []

        index_name_array.sort(key=lambda v:v[0])
        sorted_index = [x[0] for x in index_name_array]

        nr_of_images = len(index_name_array)
        vecs = [None] * nr_of_images
        offset = np.float32(1).nbytes * self.ndims

        res = array.array('f')
        fr = open(self.binary_file, 'rb')
        fr.seek(index_name_array[0][0] * offset)
        res.fromfile(fr, self.ndims)
        previous = index_name_array[0][0]

        for next in sorted_index[1:]:
            move = (next-1-previous) * offset
            fr.seek(move, 1)
            res.fromfile(fr, self.ndims)
            previous = next

        fr.close()

        return [x[1] for x in index_name_array], [ res[i*self.ndims:(i+1)*self.ndims].tolist() for i in range(nr_of_images) ]

    def read_one(self, name):
        renamed, vectors = self.read([name])
        return vectors[0]

    def shape(self):
        return [self.nr_of_images, self.ndims]
