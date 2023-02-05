import argparse
import os
import pickle
import numpy as np
import warnings
warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)

def ensureDir(string):
    if os.path.isdir(string):
        return string
    else:
        raise NotADirectoryError(string)


def merge_two_parsers(p1,p2):
    """
    This function is a modification of  argparse _ActionsContainer._add_container_actions
    that allows for merging duplicates 
    """
    # collect groups by titles
    title_group_map = {}
    for group in p1._action_groups:
        if group.title in title_group_map:
            raise ValueError(f'cannot merge actions - two groups are named {group.title}')
        title_group_map[group.title] = group

    # map each action to its group
    group_map = {}
    for group in p2._action_groups:

        # if a group with the title exists, use that, otherwise
        # create a new group matching the p2's group
        if group.title not in title_group_map:
            title_group_map[group.title] = p1.add_argument_group(
                title=group.title,
                description=group.description,
                conflict_handler=group.conflict_handler)

        # map the actions to their new group
        for action in group._group_actions:
            group_map[action] = title_group_map[group.title]

    # add p2's mutually exclusive groups
    # NOTE: if add_mutually_exclusive_group ever gains title= and
    # description= then this code will need to be expanded as above
    for group in p2._mutually_exclusive_groups:
        mutex_group = p1.add_mutually_exclusive_group(
            required=group.required)

        # map the actions to their new mutex group
        for action in group._group_actions:
            group_map[action] = mutex_group
    # add all actions to this p2 or their group
    for action in p2._actions:
        try:
            group_map.get(action, p1)._add_action(action)
        except:
            print("Warning:", action.option_strings, "is a duplicate" )
    return p1


def get_state_filename(save_file_name: str) -> str:
    """
    :return: evolution save filename according to the used convention, None if the input parameter is None
    """
    return None if save_file_name is None else save_file_name + '_state.pkl'


def get_state_from_state_file(state_filename: str):
    """
    Gets evolution state from saved state file

    :param state_filename: name of the file storing the saved evolution state
    :return: state object of the saved evolution, None otherwise
    """
    if state_filename is None:
        return None
    try:
        with open(state_filename, 'rb') as f:
            state = pickle.load(f)
    except FileNotFoundError:
        return None

    return state


def get_evolution_from_state_file(hof_savefile: str):
    """
    Gets evolution state from the respective savefile, according to the followed convention of naming state savefiles

    :param hof_savefile: filename, which will be converted according to the followed convention to look for respective
    evolution state file
    :return: evolution state object if save file found, None otherwise
    """
    state_filename = get_state_filename(hof_savefile)
    return get_state_from_state_file(state_filename)


def write_state_to_file(state, state_filename: str):
    """
    Writes the state to the respective state file

    :param state: state object
    :param state_filename: name of the file to write to
    """
    state_filename_tmp = f'{state_filename}.tmp'
    try:
        with open(state_filename_tmp, "wb") as f:
            pickle.dump(state, f)
        # ensures the new file was first saved OK (e.g. enough free space on device), then replace
        os.replace(state_filename_tmp, state_filename)
    except Exception as ex:
        raise RuntimeError(
            f"Exception: Failed to save evolution state {state_filename_tmp}\n"
            f"Message: {ex}\n"
            f"This does not prevent the experiment from continuing, but let\'s stop here to fix the problem with saving state files."
        )


def evaluate_cec2017(genotype, cec_benchmark_function):
    if any(x < -100 or x > 100 for x in genotype):
        return -np.inf
    cec2017_genotype = np.array([genotype])
    return cec_benchmark_function(cec2017_genotype)
