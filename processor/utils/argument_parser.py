import argparse
import os
import sys

import yaml


class ArgumentParser:
    def __init__(self, argv):
        """
        Initialize argument parser:
            * Parse arguments
            * Create session file with current arguments
        :param argv: arguments from command line
        """
        self.arg = self.load_arg(argv)

        self.work_dir = self.arg.work_dir
        self.session_file = f'{self.work_dir}/config.yaml'

        if not os.path.exists(self.work_dir):
            os.makedirs(self.work_dir)
        self.save_arg(self.arg)

    def load_arg(self, argv=None):
        """
        Parse given arguments and store them
        :param argv: arguments from command line
        :return: parsed arguments
        """
        parser = self.get_parser()

        # load arg from config file
        p = parser.parse_args(argv)
        if p.config is not None:
            # load config file
            with open(p.config, 'r') as f:
                default_arg = yaml.load(f)

            # update parser from config file
            key = vars(p).keys()
            for k in default_arg.keys():
                if k not in key:
                    print(f'Unknown Arguments: {k}')
                    assert k in key

            parser.set_defaults(**default_arg)

        return parser.parse_args(argv)

    def save_arg(self, arg):
        """
        Save given arguments at predefined sessions file
        :param arg: arguments to save
        """
        arg_dict = vars(arg)
        with open(self.session_file, 'w') as f:
            f.write(f'# command line: {" ".join(sys.argv)}\n\n')
            yaml.dump(arg_dict, f, default_flow_style=False, indent=4)

    @staticmethod
    def get_parser(add_help=True):
        """
        Create basic argument parser
        :param add_help: boolean flag to enable command line help
        :return: parser
        """
        # parameter priority: command line > config > default
        parser = argparse.ArgumentParser(add_help=add_help, description='Argument Parser')

        parser.add_argument('-w', '--work_dir', default='./work_dir/tmp', help='the work folder for storing results')
        parser.add_argument('-c', '--config', default=None, help='path to the configuration file')

        return parser
