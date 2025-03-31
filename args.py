import argparse


def parameter_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name', type=str, default='CICIDS', choices=["TONIoT","DoHBrw", "CICIDS", "CICMalMen"],
                help='which dataset to use')
    parser.add_argument("--cuda", type=str, default='1', help="Device: cuda:num or cpu.")
    parser.add_argument("--config", action='store_true', default=True, help="Read configuration file.")

    parser.add_argument('--n_class', dest='n_class', type=int)
    parser.add_argument('--b', dest='binary', action='store_true',
                         default=False, help='True if you want binary classification.')
    
    parser.add_argument('--lr', dest='lr', type=float, default=1e-3)
    parser.add_argument('--dropout', type=float, default=0.5,
                        help='Dropout rate (1 - keep probability).')
    parser.add_argument('--weight_decay', type=float, default=5e-4,
                        help='Weight decay (L2 loss on parameters).')

    parser.add_argument('--train_ratio', type=float, default=0.3)
    parser.add_argument('--epo', dest='epochs', type=int, default=500)
    parser.add_argument("--annealing_epoch", type=int, default=5)
    parser.add_argument('--batch', dest='batch_size', type=int, default=16)
    parser.add_argument('--texthead', dest='texthead', type=int, default=200)
    parser.add_argument('--run', dest='run', type=int, default=1)
    parser.add_argument('--patience', type=int, default=500)
    parser.add_argument('--hid', dest='hid', type=int, default=64)

    args = parser.parse_args()

    return args
