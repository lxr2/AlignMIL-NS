import argparse
from omegaconf import OmegaConf
from utils import init, print_config
from manager import Manager
from datetime import datetime


def main(config):
    print_config(config)
    manager = Manager(config) # Operation Class
    manager.incre_train() # incremental training
    print('[infor] end of program execution')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', '-f', type=str, required=True, help='path to the YAML config file')
    args, remaining = parser.parse_known_args()

    conf = OmegaConf.create({'config': args.config})
    yaml_conf = OmegaConf.load(args.config)
    cli_conf = OmegaConf.from_dotlist(remaining)
    conf = OmegaConf.merge(conf, yaml_conf, cli_conf)

    if 'name' not in conf:
        conf['name'] = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

    init(conf)
    main(conf)