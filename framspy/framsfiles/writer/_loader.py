import json
import warnings

from framsfiles._context import _contexts
from ._parser import _parse_object_list, _parse_object

_NO_FILE_EXTENSION_WARNING = 'No file extension found. Setting default context.'
_UNSUPPORTED_EXTENSION_WARNING = 'Unsupported file extension: \'{}\'. Setting default context.'
_UNSUPPORTED_CONTEXT_WARNING = 'Unsupported context: "{}". Setting default context.'

_INVALID_ROOT_ERROR = 'JSON root should be an object or a list, found: {}. Aborting.'


def from_file(filename, context=None):
    """
    Converts the file with a given filename to Framsticks file format.
    :param filename: Name of the file to parse.
    :param context: Context of parsing compliant with contexts found in 'framscript.xml' e.g. 'expdef file'.
    If context is left empty it will be inferred from the file's extension.
    :return: String content of equivalent Framsticks file.
    """
    if context is None:
        context = _get_context_from_filename(filename)
    with open(filename, encoding='UTF-8') as file:
        json_object = json.load(file)
    return from_collection(json_object, context)


def from_collection(target, context=None):
    """
    Converts a list or a dictionary to Framsticks file format.
    :param target: Dictionary or list of dictionaries representing Framsticks objects.
    :param context: Context of parsing compliant with contexts found in 'framscript.xml'
    :return: String content of resulting Framsticks file.
    """
    if context is not None and context not in _contexts:
        warnings.warn(_UNSUPPORTED_CONTEXT_WARNING.format(context))
    if isinstance(target, list):
        return _parse_object_list(target, context)
    if isinstance(target, dict):
        return _parse_object(target, context)
    raise ValueError(_INVALID_ROOT_ERROR.format(type(target)))


def _get_context_from_filename(filename):
    filename_split = filename.split('.')
    if len(filename_split) < 2:
        warnings.warn(_NO_FILE_EXTENSION_WARNING)
        return None

    extension = filename_split[-2]
    filename_context = extension + ' file'
    if filename_context not in _contexts:
        warnings.warn(_UNSUPPORTED_EXTENSION_WARNING.format(filename_context))
        return None

    return filename_context
