import argparse


def parameter_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name', type=str, default='CICIDS', choices=["TONIoT","DoHBrw", "CICIDS", "CICMalMen"],
                help='which dataset to use')
    parser.add_argument("--cuda", type=str, default='0', help="Device: cuda:num or cpu.")
    parser.add_argument('--emb', dest= 'embedding_type',type=str, default='all', help='embedding type', choices=["mpnet", "par", "all"])
    parser.add_argument('--pretrain', dest= 'pretrain_model', type=str, default='qwen', help='pretrain type', choices=["qwen3","gpt3", "qwen", "mistral","zephyr","gemma"])
    parser.add_argument('--run_id', type=str, default='', help='Unique run identifier for parallel grid search')
    parser.add_argument('--no_test', action='store_true', default=False, help='Skip test phase')
    parser.add_argument('--evl', default=True)
    parser.add_argument('--views', type=str, default='0,1,2', help='Comma-separated list of views to use (e.g., "1,2" or "1,2,3")')
    parser.add_argument("--seed", default='0', type=int)
    parser.add_argument('--n_class', dest='n_class', type=int)


    parser.add_argument('--lr', dest='lr', type=float, default=1e-4)
    parser.add_argument('--dropout', type=float, default=0.5,
                        help='Dropout rate (1 - keep probability).')
    parser.add_argument('--weight_decay', type=float, default=5e-4,
                        help='Weight decay (L2 loss on parameters).')
    parser.add_argument('--dc_loss', type=float, default=1.0,
                        help='Weight for the DC loss component.')
    
    parser.add_argument('--train_ratio', type=float, default=0.1)
    parser.add_argument('--epo', dest='epochs', type=int, default=3000)
    parser.add_argument("--annealing_epoch", type=int, default=15)
    # parser.add_argument('--batch', dest='batch_size', type=int, default=16)
    # parser.add_argument('--texthead', dest='texthead', type=int, default=1000)
    parser.add_argument('--run', dest='run', type=int, default=1)
    parser.add_argument('--patience', type=int, default=2000)
    parser.add_argument('--hid', dest='hid', type=int, default=64)

    args = parser.parse_args()

    return args
