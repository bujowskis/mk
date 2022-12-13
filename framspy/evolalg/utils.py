import argparse
import os


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
            msg = ('cannot merge actions - two groups are named %r')
            raise ValueError(msg % (group.title))
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
            print("Warning:",action.option_strings, "is a duplicate" )
    return p1