import argparse, sys, os

from prakarsh.human_classification import (human_class)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser = argparse.ArgumentParser(description='Human classification')

    parser.add_argument('--data_dir',default='DEAP/data', type=str,
                        help= "Data directory containing files")
    
    args = parser.parse_args(sys.argv[1:])
    human_class(args)