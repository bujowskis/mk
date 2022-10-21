import warnings

from framsfiles._context import _specs
from ._serializer import _serialize_value

_EXPECTED_OBJECT_WARNING = 'Encountered item of type {} in list of objects. Skipping.'
_MISSING_CLASSNAME_WARNING = 'Object defined without classname. Resulting file might be invalid.'
_INVALID_TYPE_WARINING = 'Field {} in class {} has type {}, {} expected.'
_LOWER_LIMIT_WARINING = 'Value {} of field {} in class {} is less than {}.'
_UPPER_LIMIT_WARINING = 'Value {} of field {} in class {} is more than {}.'

_CAST_ERROR = 'Cannot cast {} to {}, aborting.'


def _parse_object_list(object_list, context=None):
    fram_objects = []
    for obj in object_list:
        if isinstance(obj, dict):
            fram_objects.append(_parse_object(obj, context))
        else:
            warnings.warn(_EXPECTED_OBJECT_WARNING.format(type(obj)))
    return '\n'.join(fram_objects)


def _parse_object(obj, context=None):
    line_list = []
    spec = None
    classname = obj.get('_classname')

    if classname is None:
        warnings.warn(_MISSING_CLASSNAME_WARNING)
    else:
        line_list = [classname + ':']
        spec = _specs.get((context, classname))

    for key, value in obj.items():
        if not _is_classname(key):
            if spec is not None:
                _validate_field(key, value, classname, spec)
            if isinstance(value, str):
                if _is_multiline(value):
                    value = _parse_multiline(value)
                elif _contains_serialized_keyword(value) or _contains_tab(value):
                    value = _serialize_value(value)
            elif isinstance(value, (list, dict)):
                value = _serialize_value(value)
            line = _to_fram_field_string(key, value)
            line_list.append(line)

    return '\n'.join(line_list) + '\n'


def _validate_field(key, value, classname, spec):
    _validate_type(key, value, classname, spec)
    _validate_min(key, value, classname, spec)
    _validate_max(key, value, classname, spec)


def _validate_type(key, value, classname, spec):
    if not isinstance(value, spec['dtype']):
        warnings.warn(_INVALID_TYPE_WARINING.format(key, classname, type(value), spec['type']))


def _validate_min(key, value, classname, spec):
    if 'min' in spec:
        casted = spec['dtype'](value)
        if casted < spec['min']:
            warnings.warn(_LOWER_LIMIT_WARINING.format(value, key, classname, spec['min']))


def _validate_max(key, value, classname, spec):
    if 'max' in spec:
        casted = spec['dtype'](value)
        if casted > spec['max']:
            warnings.warn(_UPPER_LIMIT_WARINING.format(value, key, classname, spec['max']))


def _is_classname(key):
    return key == '_classname'


def _parse_multiline(value):
    value = _escape_tildes(value)
    return _tilde_wrap(value)


def _escape_tildes(value):
    return value.replace('~', '\\~')


def _tilde_wrap(value):
    return '~\n{}~'.format(value)


def _to_fram_field_string(key, value):
    return '{}:{}'.format(key, value)


def _contains_serialized_keyword(value):
    return '@Serialized:' in value


def _contains_tab(value):
    return '\t' in value


def _is_multiline(value):
    return '\n' in value
