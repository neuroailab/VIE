import os

MIT_ROOT = '/data/vision/oliva/scratch/aandonia/moments/models/datasets/kinetics'
MIT_ROOT_DATA = '/data/vision/oliva/scratch/datasets/kinetics/comp_jpgs_extracted'


def return_kinetics(root=MIT_ROOT, root_data=MIT_ROOT_DATA):
    """Return the split information."""
    filename_categories = os.path.join(root, 'categories.txt')
    filename_imglist_train = os.path.join(root, 'train_frameno_new.txt')
    filename_imglist_val = os.path.join(root, 'val_frameno_new.txt')
    prefix = '{:06d}.jpg'
    return filename_categories, filename_imglist_train, filename_imglist_val, root_data, prefix


def return_moments():
    filename_categories = '/data/vision/oliva/scratch/moments/split/categoryList_nov17.csv'
    prefix = '{:06d}.jpg'
    root_data = '/data/vision/oliva/scratch/moments/moments_nov17_frames'
    filename_imglist_train = '/data/vision/oliva/scratch/moments/split/rgb_trainingSet_nov17.csv'
    filename_imglist_val = '/data/vision/oliva/scratch/moments/split/rgb_validationSet_nov17.csv'
    return filename_categories, filename_imglist_train, filename_imglist_val, root_data, prefix


def return_UCF101(root=MIT_ROOT, root_data=MIT_ROOT_DATA):
    filename_categories = os.path.join(root, 'categories.txt')
    filename_imglist_train = os.path.join(root, 'trainlist01_meta.txt')
    filename_imglist_val = os.path.join(root, 'testlist01_meta.txt')
    prefix = '{:06d}.jpg'
    return filename_categories, filename_imglist_train, filename_imglist_val, root_data, prefix

def return_UCF101_2(root=MIT_ROOT, root_data=MIT_ROOT_DATA):
    filename_categories = os.path.join(root, 'categories.txt')
    filename_imglist_train = os.path.join(root, 'trainlist02_meta.txt')
    filename_imglist_val = os.path.join(root, 'testlist02_meta.txt')
    prefix = '{:06d}.jpg'
    return filename_categories, filename_imglist_train, filename_imglist_val, root_data, prefix

def return_UCF101_3(root=MIT_ROOT, root_data=MIT_ROOT_DATA):
    filename_categories = os.path.join(root, 'categories.txt')
    filename_imglist_train = os.path.join(root, 'trainlist03_meta.txt')
    filename_imglist_val = os.path.join(root, 'testlist03_meta.txt')
    prefix = '{:06d}.jpg'
    return filename_categories, filename_imglist_train, filename_imglist_val, root_data, prefix


def return_HMDB51(root=MIT_ROOT, root_data=MIT_ROOT_DATA):
    filename_categories = os.path.join(root, 'categories.txt')
    filename_imglist_train = os.path.join(root, 'trainlist01_meta.txt')
    filename_imglist_val = os.path.join(root, 'testlist01_meta.txt')
    prefix = '{:06d}.jpg'
    return filename_categories, filename_imglist_train, filename_imglist_val, root_data, prefix

def return_HMDB51_2(root=MIT_ROOT, root_data=MIT_ROOT_DATA):
    filename_categories = os.path.join(root, 'categories.txt')
    filename_imglist_train = os.path.join(root, 'trainlist02_meta.txt')
    filename_imglist_val = os.path.join(root, 'testlist02_meta.txt')
    prefix = '{:06d}.jpg'
    return filename_categories, filename_imglist_train, filename_imglist_val, root_data, prefix

def return_HMDB51_3(root=MIT_ROOT, root_data=MIT_ROOT_DATA):
    filename_categories = os.path.join(root, 'categories.txt')
    filename_imglist_train = os.path.join(root, 'trainlist03_meta.txt')
    filename_imglist_val = os.path.join(root, 'testlist03_meta.txt')
    prefix = '{:06d}.jpg'
    return filename_categories, filename_imglist_train, filename_imglist_val, root_data, prefix

def return_infant(root, root_data):
    filename_imglist_val = os.path.join(root, "infant_30min_metafile.txt")
    return filename_imglist_val, root_data


def dataset_config(dataset, **kwargs):
    datasets = {
        'hmdb0': {},
        'hmdb1': {},
        'hmdb2': {},
        'ucf101': {},
        'jester': {},
        'charades': {},
        'something': {},
        'somethingv2': {},
        'moments': return_moments,
        'kinetics': return_kinetics,
        'UCF101': return_UCF101,
        'UCF101_2': return_UCF101_2,
        'UCF101_3': return_UCF101_3,
        'HMDB51': return_HMDB51,
        'HMDB51_2': return_HMDB51_2,
        'HMDB51_3': return_HMDB51_3,}
    
    if dataset == 'infant':
        file_imglist_val, root_data = return_infant(**kwargs)
        return {
            'val_metafile': file_imglist_val,
            'root': root_data
        }

    if dataset in datasets:
        file_categories, file_imglist_train, \
                file_imglist_val, root_data, \
                prefix = datasets[dataset](**kwargs)
    else:
        raise ValueError('Unknown dataset {}'.format(dataset))

    with open(file_categories) as f:
        categories = [line.rstrip() for line in f.readlines()]

    return {
        'categories': categories,
        'train_metafile': file_imglist_train,
        'val_metafile': file_imglist_val,
        'root': root_data,
        'prefix': prefix
    }
