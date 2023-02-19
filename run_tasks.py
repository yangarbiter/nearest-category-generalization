import argparse
import os
import logging

from mkdir_p import mkdir_p

from lolip.variables import auto_var, get_file_name
from params import *
from utils import setup_experiments

logging.basicConfig(level=logging.DEBUG)

DEBUG = True if os.environ.get('DEBUG', False) else False

def main(args, auto_var):
    experiments = [
        GetPreds(),
        #OutOfSampleRepr(),
        #OutOfSampleReprClf(),
        #ThreeGauss(),
        #TrainDatasetClassifier(),
        #OODRobustness(),
        #TrainDROCC(),
        #TrainVAE(),
        #fetOODRobustness(),
        #ThreeGauss(),
        ##ReprEstimateRobustness(),
        #EstimateRobustness(),
        #RandOODPred(),
        #FewShotFinetune(),
    ]
    grid_params = []
    for exp in experiments:
        exp_fn, _, grid_params, run_param = exp()

        run_param['n_jobs'] = 1
        run_param['allow_failure'] = args.allow_failure

        auto_var.run_grid_params(exp_fn, grid_params, **run_param)
    #auto_var.run_grid_params(delete_file, grid_params, n_jobs=1,
    #                          with_hook=False, allow_failure=False)

def delete_file(auto_var):
    os.unlink(get_file_name(auto_var) + '.json')

if __name__ == "__main__":
    setup_experiments(auto_var)
    parser = argparse.ArgumentParser(description='Run tasks.')
    parser.add_argument('--allow_failure', action='store_true', help='')
    args = parser.parse_args()
    main(args, auto_var)
