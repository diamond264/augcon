import argparse
import yaml
from util.config import Config
import pretrain, finetune, pretrain_backup

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True,
                        help='Path to YAML config file')
    parser.add_argument('--pretrain', action='store_true', default=False,
                        help='Run pretraining code')
    parser.add_argument('--finetune', action='store_true', default=False,
                        help='Run finetuning code')
    parser.add_argument('--debug', action='store_true', default=False,
                        help='Run debugging code')
    args = parser.parse_args()
    return args

def set_config(config, key, value):
    if isinstance(value, dict):
        for k, v in value.items():
            set_config(getattr(config, key), k, v)
    else:
        setattr(config, key, value)

def main():
    args = parse_args()

    with open(args.config, 'r') as f:
        config_dict = yaml.load(f, Loader=yaml.FullLoader)

    config = Config()
    for key, value in config_dict.items():
        set_config(config, key, value)

    if args.pretrain:
        print(f'Running pretraining code...')
        pretrain.run(config)
    elif args.finetune:
        print(f'Running finetuning code...')
        finetune.run(config)
    elif args.debug:
        print(f'Running debugging code...')
        pretrain_backup.run(config)
    else:
        print(f'Please specify either --pretrain or --finetune')

if __name__ == '__main__':
    main()
