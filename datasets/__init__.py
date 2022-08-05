from .face_folder import extract_face_folder

__datasets = {
    'face_folder': extract_face_folder,
}


def init_dataset(name, data_dir, data_func=None, verbose=True, **kwargs):
    avai_datasets = list(__datasets.keys())

    if data_func is not None:
        print("(init_dataset) INFO: use custom dataset...")
        trainset, valset, testset, attr_dict = data_func(data_dir, **kwargs)
    else:
        if name not in avai_datasets:
            raise ValueError(
                'Invalid dataset name. Received "{}", '
                'but expected to be one of {}'.format(name, avai_datasets)
            )
        trainset, valset, testset, attr_dict = __datasets[name](data_dir, **kwargs)
    if verbose:
        show_summary(trainset, valset, testset, attr_dict, name)
    
    return trainset, valset, testset, attr_dict


def show_summary(trainset, valset, testset, attr_dict, name):
    num_train = len(trainset)
    num_val = len(valset)
    num_test = len(testset)
    num_total = num_train + num_val + num_test

    print('=> Loaded {}'.format(name))
    print("  ------------------------------")
    print("  subset   | # images")
    print("  ------------------------------")
    print("  train    | {:8d}".format(num_train))
    print("  val      | {:8d}".format(num_val))
    print("  test     | {:8d}".format(num_test))
    print("  ------------------------------")
    print("  total    | {:8d}".format(num_total))
    print("  ------------------------------")
    print("  # attributes: {}".format(len(attr_dict)))
    # for label, attr in attr_dict.items():
    #     print('    {:3d}: {}'.format(label, attr))
    print("  ------------------------------")
