import argparse
import sys
from FramsticksLib import FramsticksLib

# FIXME - relative import outside package
from ..utils import ensureDir
from ..frams_base.experiment_frams_niching import ExperimentFramsNiching

# FIXME - make into smaller tests to run - this is RESEARCH

SETTINGS_TO_TEST = {
    'opt': ['velocity', 'vertpos'],
    'max_numparts': [None],
    'max_numjoints': [20],
    'max_numneurons': [20],
    'max_numconnections': [None],
    'max_numgenochars': [20],
    'hof_size': [0, 10],
    'normalize': ['none', 'max', 'sum'],
    'dissim': [-2, -1, 1, 2],
    'fit': ['niching', 'novelty', 'nsga2', 'nslc', 'raw'],
    'genformat': ['1'],
    'popsize': [8],
    'archive': [8],
    'initialgenotype': [None],
    'pmut': [0.7],
    'pxov': [0.2],
    'tournament': [5]
}


def test_run_experiment(params):
    opt_criteria = params['opt'].split(
        ",")  # multiple criteria not supported here. If needed, use FramsticksEvolution.py
    framsLib = FramsticksLib(parsed_args.path, parsed_args.lib, parsed_args.sim.split(";"))
    constrains = {
        "max_numparts": params['max_numparts'],
        "max_numjoints": params['max_numjoints'],
        "max_numneurons": params['max_numneurons'],
        "max_numconnections": params['max_numconnections'],
        "max_numgenochars": params['max_numgenochars'],
    }

    print('Best individuals:')
    experiment = ExperimentFramsNiching(
        frams_lib=framsLib,
        optimization_criteria=opt_criteria,
        hof_size=params['hof_size'],
        constraints=constrains,
        normalize=params['normalize'],
        dissim=params['dissim'],
        fit=params['fit'],
        genformat=params['genformat'],
        popsize=params['popsize'],
        archive_size=params['archive']
    )

    experiment.evolve(
        hof_savefile=None,
        generations=2,
        initialgenotype=params['initialgenotype'],
        pmut=params['pmut'],
        pxov=params['pxov'],
        tournament_size=params['tournament']
    )


def parseArguments():
    parser = argparse.ArgumentParser(
        description='Run this program with "python -u %s" if you want to disable buffering of its output.' % sys.argv[0]
    )
    parser.add_argument('-path', type=ensureDir, required=True, help='Path to Framsticks CLI without trailing slash.')
    parser.add_argument(
        '-lib',
        required=False,
        help='Library name. If not given, "frams-objects.dll" or "frams-objects.so" is assumed depending on the platform.'
    )
    parser.add_argument(
        '-sim', required=False,
        default="eval-allcriteria.sim",
        help="The name of the .sim file with settings for evaluation, mutation, crossover, and similarity estimation. If not given, \"eval-allcriteria.sim\" is assumed by default. Must be compatible with the \"standard-eval\" expdef. If you want to provide more files, separate them with a semicolon ';'."
    )

    return parser.parse_args()


def get_params_sets():
    params_sets = []
    for k in SETTINGS_TO_TEST.keys():
        temp_param_set = []
        for value in SETTINGS_TO_TEST[k]:
            if params_sets:
                for exsiting_set in params_sets:
                    copy_of_set = exsiting_set.copy()
                    copy_of_set[k] = value
                    temp_param_set.append(copy_of_set)
            else:
                temp_param_set.append({k: value})
        params_sets = temp_param_set
    return params_sets


def cover_to_test(params):
    try:
        test_run_experiment(params)
        return [1, None]
    except Exception as e:
        return [0, f"Experiment with params:{params} failied with the stack:{e}"]


def run_tests():
    params_sets = get_params_sets()
    print(f"Starting executing {len(params_sets)} experiments")
    result = [cover_to_test(params) for params in params_sets]
    Successful = [r[0] for r in result]
    print(f"{sum(Successful)} out of {len(params_sets)} passed")
    if sum(Successful) < len(params_sets):
        print("Experiments that failed failed:")
        [print(r[1]) for r in result if r[1]]


if __name__ == "__main__":
    parsed_args = parseArguments()
    run_tests()
