_SERIALIZED_STRING = '@Serialized:{}'
_DOUBLE_QUOTED = '"{}"'
_FIELD_STRING = '"{}":{}'
_LIST_STRING = '[{}]'
_OBJECT_STRING = '{{{}}}'


def _serialize_value(value):
    return _SERIALIZED_STRING.format(_serialize(value))


def _serialize(target):
    if isinstance(target, list):
        return _serialize_list(target)
    if isinstance(target, dict):
        return _serialize_object(target)
    if isinstance(target, str):
        return _serialize_string(target)
    if target is None:
        return 'null'
    return str(target)


def _serialize_list(target):
    serialized = [_serialize(el) for el in target]
    return _to_list_string(serialized)


def _serialize_object(target):
    serialized = [_to_field_string(k, v) for k, v in target.items()]
    return _to_object_string(serialized)


def _serialize_string(target):
    return _DOUBLE_QUOTED.format(target)


def _to_field_string(key, value):
    return _FIELD_STRING.format(key, _serialize(value))


def _to_list_string(target):
    return _LIST_STRING.format(','.join(target))


def _to_object_string(target):
    return _OBJECT_STRING.format(','.join(target))
