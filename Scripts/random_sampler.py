import os
import itertools
import numpy as np
import random
import math
import shutil

random.seed(100)
np.random.seed(100)

def populate(images, split_index=1):
    assert type(images) is list and len(images) > 0, "Check input..."

    cache_dict = dict()
    count_dict = dict()

    for im in images:
        split = im.split("_")[split_index]

        if split not in cache_dict.keys():
            cache_dict[split] = [i for i in images if i.startswith("UID_{}_".format(split))]
            count_dict[split] = len(cache_dict[split])

    return cache_dict, count_dict


def split(hem, all1, n_hem, n_all, ratio=0.3, tolerence=0.02):

    '''

    :param hem:
    :param all1:
    :param shuffle:
    :return:
    '''
    while True:
        random.shuffle(hem)
        random_hem = hem
        random.shuffle(all1)
        random_all = all1

        n_train_hem = int(len(hem) * 0.7)
        n_val_hem = ((n_hem - n_train_hem) // 2)
        n_test_hem = n_hem - n_val_hem - n_train_hem
        train_hem = random_hem[:n_train_hem]
        val_hem = random_hem[n_train_hem:n_train_hem+n_val_hem]
        test_hem = random_hem[-n_test_hem:]

        # train_hem = random_hem[-n_train_hem:]
        # val_hem = random_hem[-(n_train_hem+n_val_hem):-n_train_hem]
        # test_hem = random_hem[:n_test_hem]

        n_train_all = int(len(all1) * 0.7)
        n_val_all = ((n_all - n_train_all) // 2)
        n_test_all = n_all - n_val_all - n_train_all

        train_all = random_all[:n_train_all]
        val_all = random_all[n_train_all:n_train_all+n_val_all]
        test_all = random_all[-n_test_all:]

        # train_all = random_all[-n_train_all:]
        # val_all = random_all[-(n_train_all + n_val_all):-n_train_all]
        # test_all = random_all[:n_test_all]

        n_train_hem = get_sum(train_hem)
        n_val_hem = get_sum(val_hem)
        n_test_hem = get_sum(test_hem)

        n_train_all = get_sum(train_all)
        n_val_all = get_sum(val_all)
        n_test_all = get_sum(test_all)

        n_train = n_train_all + n_train_hem
        n_val = n_val_hem + n_val_all
        n_test = n_test_hem + n_test_all

        train_hem_ratio = n_train_hem / n_train
        train_all_ratio = n_train_all  / n_train
        val_hem_ratio = n_val_hem / n_val
        val_all_ratio = n_val_all / n_val
        test_hem_ratio = n_test_hem / n_test
        test_all_ratio = n_test_all / n_test

        print('Train - hem: {} | all: {}'.format(train_hem_ratio, train_all_ratio))
        print('Validation - hem: {} | all: {}'.format(val_hem_ratio, val_all_ratio))
        print('Test - hem: {} | all: {}'.format(test_hem_ratio, test_all_ratio))
        print('-------------------------------')

        if (
                (math.fabs(ratio - train_hem_ratio) <= tolerence) and
                (math.fabs(ratio - val_hem_ratio) <= tolerence) and
                (math.fabs(ratio - test_hem_ratio) <= tolerence)
        ):
            print('Done.')
            break
    return (train_hem, train_all, val_hem, val_all, test_hem, test_all)


def get_sum(samples):
    return sum([s[1] for s in samples])

def get_dist(hem_list, all_list):
    hem_samples = get_sum(hem_list)
    all_samples = get_sum(all_list)
    total_samples = hem_samples + all_samples
    return 'HEM: {} | ALL: {}'.format(hem_samples/total_samples, all_samples/total_samples)


if __name__ == "__main__":

    # Change the working dir
    os.chdir(r'D:\Personal\ISBI\train - Copy')

    folds = ['fold_0', 'fold_1', 'fold_2']
    ALL_IMAGES = [os.listdir(os.path.join(os.getcwd(), fold, 'all')) for fold in folds]
    HEM_IMAGES = [os.listdir(os.path.join(os.getcwd(), fold, 'hem')) for fold in folds]

    ALL = []
    HEM = []

    for _all in ALL_IMAGES:
        ALL += _all

    for hem in HEM_IMAGES:
        HEM += hem

    num_hem = len(ALL)
    num_all = len(HEM)
    hem_dist = num_hem/(num_hem + num_all)
    all_dist = num_all/(num_hem + num_all)

    print('HEM:', hem_dist, ',', 'ALL:', all_dist)

    hem_dict, count_hem = populate(HEM)
    all_dict, count_all = populate(ALL)

    n_hem_g = len(count_hem)
    n_all_g = len(count_all)

    # Sort the distribution in the ascending order of the number of images associated with each ID.
    sorted_hem = sorted(count_hem.items(), key=lambda x: x[1])
    sorted_all = sorted(count_all.items(), key=lambda x: x[1])

    # sorted_hem = [(k, v) for k, v in count_hem.items()]
    # sorted_all = [(k, v) for k, v in count_all.items()]

    (train_hem, train_all, val_hem,
     val_all, test_hem, test_all) = split(sorted_hem, sorted_all, n_hem_g, n_all_g, tolerence=0.02)
    print(train_hem)
    print(train_all)
    print(val_hem)
    print(val_all)
    print(test_hem)
    print(test_all)

    print(get_sum(train_hem))
    print(get_sum(train_all))
    print(get_sum(val_hem))
    print(get_sum(val_all))
    print(get_sum(test_hem))
    print(get_sum(test_all))

    # sets = ['train', 'val', 'test']
    # classes = ['all', 'hem']
    #
    # uids = {
    #     'train': {
    #         'all': train_all,
    #         'hem': train_hem
    #     },
    #
    #     'val': {
    #         'all': val_all,
    #         'hem': val_hem
    #     },
    #
    #     'test': {
    #         'all': test_all,
    #         'hem': test_hem
    #     }
    # }
    #
    # if not os.path.exists(r'..\final_data'):
    #     os.mkdir(r'..\final_data')
    # for _set in sets:
    #     print('Copying files: {}'.format(_set))
    #     root = os.path.join(r'..\final_data', _set)
    #     if not os.path.exists(root):
    #         os.mkdir(root)
    #     for _class in classes:
    #         print('\tCopying files: {}'.format(_class))
    #         node = os.path.join(root, _class)
    #         if not os.path.exists(node):
    #             os.mkdir(node)
    #         _uids = [u[0] for u in uids[_set][_class]]
    #         filenames = []
    #         for fold in folds:
    #             filepath = os.path.join(os.getcwd(), fold, _class)
    #             files = os.listdir(filepath)
    #             for _file in files:
    #                 if any(['UID_{}_'.format(_uid) in _file for _uid in _uids]):
    #                     src = os.path.join(filepath, _file)
    #                     dst = os.path.join(node, _file)
    #                     shutil.copy(src, dst)
    #         print('\tDone.')

