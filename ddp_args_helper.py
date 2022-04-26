import argparse
import sys
import yaml

from configs import parser as _parser

global parser_args

class ArgsHelper:
    def parse_arguments(self, jupyter_mode=False):
        parser = argparse.ArgumentParser(description="Pruning random networks")

        # Config/Hyperparameters
        parser.add_argument(
            "--gpu",
            type=int,
            default=0,
            help="gpu"
        )
        parser.add_argument(
            "--name",
            default="blah",
            type=str,
            help="Name of experiment"
        )
        
        if jupyter_mode:
            args = parser.parse_args("")
        else:
            args = parser.parse_args()

        return args

    def isNotebook(self):
        try:
            shell = get_ipython().__class__.__name__
            if shell == 'ZMQInteractiveShell':
                return True   # Jupyter notebook or qtconsole
            elif shell == 'TerminalInteractiveShell':
                return False  # Terminal running IPython
            else:
                return False  # Other type (?)
        except NameError:
            return False      # Probably standard Python interpreter

    def get_args(self, jupyter_mode=False):
        global parser_args
        jupyter_mode = self.isNotebook()
        parser_args = self.parse_arguments(jupyter_mode)

argshelper = ArgsHelper()
argshelper.get_args()
