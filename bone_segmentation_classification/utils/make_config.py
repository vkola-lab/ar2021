import configparser, ast
from optparse import OptionParser


def args_train():  # get arguments
    parser = OptionParser()
    parser.add_option('-e', '--epochs', dest='epochs', default=200, type='int',
                      help='number of epochs')
    parser.add_option('-b', '--batch-size', dest='batch_size', default=16,
                      type='int', help='batch size')
    parser.add_option('-l', '--learning-rate', dest='lr', default=0.02,
                      type='float', help='learning rate')
    parser.add_option('-g', '--gpu', action='store_true', dest='gpu',
                      default=True, help='use cuda'),
    parser.add_option('--sv', '--save', dest='save_cp',
                      default=False, help='save model parameters')
    parser.add_option('-c', '--load', dest='load',
                      default=False, help='load saved model parameters')
    parser.add_option('-s', '--scale', dest='scale', type='float',
                      default=1.0, help='downscaling factor of the images')

    (options, args) = parser.parse_args()
    return options


def write_config(name, args):
    config = configparser.ConfigParser()
    for k in args.keys():
        config[k] = args[k]

    with open(name, 'w') as configfile:
        config.write(configfile)


def load_config(file_name='config/default.ini', section='train'):
    config = configparser.ConfigParser()
    config.read(file_name)
    config.sections()

    d = dict(config._sections[section])
    for k in d.keys():
        try:
            d[k] = ast.literal_eval(d[k])
        except:
            d[k] = d[k]
    return d


if __name__ == "__main__":
    args = args_train()
    args_d = {'mask_name': 'bone_resize_B_patch_2',
              'mask_used': [[2]],  # [[1], [2, 3], [5, 6]],
              'thickness': 1,
              'copy_channel': True,
              'pick': False,
              'elastic_deformation': False,
              'method': 'automatic'}

    write_config('config/default.ini', {'train': vars(args), 'data': args_d})




