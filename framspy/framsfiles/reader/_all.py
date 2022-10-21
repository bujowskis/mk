import os.path
import re as _re
import warnings

from framsfiles._context import _create_specs_from_xml

warnings.simplefilter('always', UserWarning)

_INT_FLOAT_REGEX = r'([+|-]?(?:0|[1-9]\d*))(\.\d+)?([eE][-+]?\d+)?'
_NATURAL_REGEX = r'(?:0|[1-9]\d*)'
_HEX__NUMBER_REGEX = r'[+|-]?0[xX][\da-fA-F]*'
_NUMBER_REGEX = '({}|{})'.format(_HEX__NUMBER_REGEX, _INT_FLOAT_REGEX)
_TYLDA_REGEX = '(?<![\\\\])(~)'
_QUOTE_REGEX = '(?<![\\\\])(")'
_ESCAPED_QUOTE_REGEX = '\\\\"'
_ESCAPED_TAB_REGEX = '\\\\t'
_ESCAPED_NEWLINE_REGEX = '\\\\n'
_ESCAPED_TYLDA_REGEX = '\\\\~'
_FRAMSCRIPT_XML_PATH = os.path.join((os.path.dirname(__file__)), "framscript.xml")

# Messages:
_NO_FILE_EXTENSION_WARNING = "No file extension found. Setting default context."
_UNSUPPORTED_EXTENSION_WARNING = "Unsupported file extension: '{}'. Setting default context."
_UNSUPPORTED_CONTEXT_WARNING = "Unsupported context: '{}'. Setting default context."
_UNEXPECTED_KEY_WARNING = "Unexpected key encountered: key: '{}', class: '{}', context: '{}' )"
_NOT_A_NUMBER_ERROR = "Expression cannot be parsed to a number: {}"
_MULTILINE_NOT_CLOSED_WARNING = "Multiline property for key: '{}' was not closed with '~'."
_STRING_NOT_CLOSED_ERROR = "String expression not closed with '~'"
_EMPTY_SERIALIZED_ERROR = "Empty value for '@Serialized' not allowed."
_NO_OBJECT_ERROR = "No object defined for the current line."
_XYZ_ERROR = "XYZ format should look like this: XYZ[ a,b,c], Got: '{}'"
_REFERENCE_FORMAT_ERROR = "reference sign '^' should be followed by an integer. Got: {}"
_COLON_EXPECTED_ERROR = "Colon ':' was expected. Got: {}"
_MIN_VAL_EXCEEDED_ERROR = "Minimum value allowed: {}, got: {}"
_MAX_VAL_EXCEEDED_ERROR = "Maximum value allowed: {}, got: {}"
_NONEMPTY_CLASSNAME = "There should be no string after obejct's classname."


_specs, _contexts = _create_specs_from_xml()


def _create_generic_parser(dtype, min=None, max=None):
    def parse(x):
        x = dtype(x)
        if min is not None:
            if x < min:
                raise ValueError(_MIN_VAL_EXCEEDED_ERROR.format(min, x))
        if max is not None:
            if x > max:
                raise ValueError(_MAX_VAL_EXCEEDED_ERROR.format(max, x))
        return x

    return parse


def _str_to_number(s):
    assert isinstance(s, str)
    s = s.strip()

    try:
        parsed_int = int(s, 0)
        return parsed_int
    except ValueError:
        pass
    try:
        parsed_float = float(s)
        return parsed_float

    except ValueError:
        pass
    raise ValueError(_NOT_A_NUMBER_ERROR.format(s))


def parse_value(value, classname=None, key=None, context=None, autoparse=True):
    assert isinstance(value, str)
    value = value.strip()
    # TODO maybe check 'Global context' as well?
    if (context, classname) in _specs:
        spec = _specs[(context, classname)]
        if key in spec:
            parser = _create_generic_parser(**spec[key])
            return parser(value)
        else:
            warnings.warn(_UNEXPECTED_KEY_WARNING.format(key, classname, context))

    if value.startswith("@Serialized:"):
        prop = value.split(":", 1)[1]
        prop = deserialize(prop)
        return prop
    elif autoparse:
        try:
            parsed_number = _str_to_number(value)
            return parsed_number
        except ValueError:
            pass
    return value


def _extract_string(exp):
    exp = exp[1:]
    str_end_match = _re.search(_QUOTE_REGEX, exp)
    if str_end_match is None:
        raise ValueError(_STRING_NOT_CLOSED_ERROR.format(exp))
    str_end = str_end_match.span()[0]
    s = exp[:str_end]
    reminder = exp[str_end + 1:]
    s = _re.sub(_ESCAPED_QUOTE_REGEX, '"', s)
    s = _re.sub(_ESCAPED_TAB_REGEX, '\t', s)
    s = _re.sub(_ESCAPED_NEWLINE_REGEX, '\n', s)
    return s, reminder


def _extract_number(exp):
    match = _re.match(_NUMBER_REGEX, exp)
    number_as_str = match.group()
    reminder = exp[match.span()[1]:]
    number = _str_to_number(number_as_str)
    return number, reminder


# TODO maybe do it nicer??
def _extract_xyz(exp):
    exp = exp.strip()
    if not exp.startswith('XYZ['):
        raise ValueError(_XYZ_ERROR.format(exp))
    exp = exp[4:]
    x, exp = _extract_number(exp)
    x = float(x)
    exp = exp.strip()
    if exp[0] != ',':
        raise ValueError(_XYZ_ERROR.format(exp))
    exp = exp[1:]
    y, exp = _extract_number(exp)
    y = float(y)
    exp = exp.strip()
    if exp[0] != ',':
        raise ValueError(_XYZ_ERROR.format(exp))
    exp = exp[1:]
    z, exp = _extract_number(exp)
    z = float(z)
    exp = exp.strip()
    if exp[0] != ']':
        raise ValueError(_XYZ_ERROR.format(exp))
    return (x, y, z), exp[1:]


def _extract_reference(exp):
    exp = exp[1:].strip()
    i_match = _re.match(_NATURAL_REGEX, exp)
    if i_match is None:
        raise ValueError(_REFERENCE_FORMAT_ERROR.format(exp))
    else:
        end_i = i_match.span()[1]
        ref_index = int(exp[:end_i])
        reminder = exp[end_i:]
    return ref_index, reminder


def _extract_custom_object(exp):
    open_braces = 0
    open_sbrackets = 0
    open_pbrackets = 0
    # TODO maybe do it smarter?
    suffix_end_match = _re.search('<|\[|\{]', exp)
    if suffix_end_match is None:
        # TODO
        raise ValueError()

    suffix_end_i = suffix_end_match.span()[0]
    i = 0
    for i, c in enumerate(exp[suffix_end_i:], start=suffix_end_i):
        if c == '<':
            open_pbrackets += 1
        elif c == '[':
            open_sbrackets += 1
        elif c == '{':
            open_braces += 1
        elif c == '>':
            open_pbrackets -= 1
        elif c == ']':
            open_sbrackets -= 1
        elif c == '}':
            open_braces -= 1

        if open_braces == 0 and open_sbrackets == 0 and open_pbrackets == 0:
            break
    if open_braces != 0 or open_sbrackets != 0 or open_pbrackets != 0:
        # TODO
        raise ValueError()
    return exp[0:i + 1], exp[i + 1:]


def deserialize(expression):
    stripped_exp = expression.strip()
    if stripped_exp == '':
        raise ValueError(_EMPTY_SERIALIZED_ERROR)
    # Just load with json ...

    if stripped_exp == 'null':
        return None

    objects = []
    references = []
    main_object_determined = False
    main_object = None
    expect_dict_value = False
    last_dict_key = None
    exp = stripped_exp
    opened_lists = 0
    opened_dicts = 0

    while len(exp) > 0:
        current_object_is_reference = False
        if main_object_determined and len(objects) == 0:
            raise ValueError(_NO_OBJECT_ERROR)
        if expect_dict_value:
            if exp[0] == ':':
                exp = exp[1:].strip()
            else:
                raise ValueError(_COLON_EXPECTED_ERROR.foramt(exp[0]))
        # List continuation
        # TODO support for XYZ tuples
        if exp[0] == ",":
            if not (isinstance(objects[-1], list) or (isinstance(objects[-1], dict) and not expect_dict_value)):
                # TODO msg
                raise ValueError()
            else:
                exp = exp[1:].strip()

        if exp[0] == "]":
            if not isinstance(objects[-1], list):
                # TODO msg
                raise ValueError()
            else:
                opened_lists -= 1
                objects.pop()
                exp = exp[1:].strip()
                continue
        elif exp[0] == "}":
            opened_dicts -= 1
            if not isinstance(objects[-1], dict):
                # TODO msg
                raise ValueError()
            else:
                objects.pop()
                exp = exp[1:].strip()
                continue
        # List start
        elif exp.startswith("null"):
            current_object = None
            exp = exp[4:]
        elif exp.startswith("XYZ"):
            current_object, exp = _extract_xyz(exp)
        elif exp[0] == "[":
            current_object = list()
            opened_lists += 1
            exp = exp[1:]
        elif exp[0] == "{":
            current_object = dict()
            opened_dicts += 1
            exp = exp[1:]
        elif exp[0] == '"':
            current_object, exp = _extract_string(exp)
        elif _re.match(_NUMBER_REGEX, exp) is not None:
            current_object, exp = _extract_number(exp)
        elif exp[0] == '^':
            i, exp = _extract_reference(exp)
            if i >= len(references):
                # TODO msg
                raise ValueError()
            current_object = references[i]
            current_object_is_reference = True
        else:
            current_object, exp = _extract_custom_object(exp)

        if len(objects) > 0:
            if isinstance(objects[-1], list):
                objects[-1].append(current_object)
            elif isinstance(objects[-1], dict):
                if expect_dict_value:
                    objects[-1][last_dict_key] = current_object
                    last_dict_key = None
                    expect_dict_value = False
                else:
                    if not isinstance(current_object, str):
                        # TODO msg
                        raise ValueError()
                    last_dict_key = current_object
                    expect_dict_value = True

        if isinstance(current_object, (list, dict, tuple)) and not current_object_is_reference:
            objects.append(current_object)
            references.append(current_object)
        if not main_object_determined:
            main_object_determined = True
            main_object = current_object
        exp = exp.strip()

    if opened_lists != 0:
        # TODO msg
        raise ValueError()
    if opened_dicts != 0:
        # TODO msg
        raise ValueError()
    return main_object


def loads(input_string, context=None, autocast=True):
    """
    Parses string in Framsticks' format to a list of dictionaries.
    :param input_string: String to parse.
    :param context: Context of parsing compliant with contexts found in 'framscript.xml' e.g. 'expdef file'.
    :param autocast: If true numbers will be parsed automatically if possible.
    If false every field will be treated as a string.
    :return: A list of dictionaries representing Framsticks objects.
    """
    assert isinstance(input_string, str)
    if context is not None and context not in _contexts:
        warnings.warn(_UNSUPPORTED_CONTEXT_WARNING.format(context))

    lines = input_string.split("\n")
    multiline_value = None
    multiline_key = None
    current_object = None
    objects = []
    parsing_error = False
    class_name = None
    try:
        for line_num, line in enumerate(lines):

            if multiline_key is not None:
                endmatch = _re.search(_TYLDA_REGEX, line)
                if endmatch is not None:
                    endi = endmatch.span()[0]
                    value = line[0:endi]
                    reminder = line[endi + 1:].strip()
                    if reminder != "":
                        # TODO msg
                        raise ValueError()
                else:
                    value = line + "\n"

                if _re.search(_TYLDA_REGEX, value) is not None:
                    # TODO msg
                    raise ValueError()
                value = _re.sub(_ESCAPED_TYLDA_REGEX, '~', value)
                multiline_value += value
                if endmatch is not None:
                    current_object[multiline_key] = multiline_value
                    multiline_value = None
                    multiline_key = None

            # Ignores comment lines (if outside multiline prop)
            elif line.startswith("#"):
                continue
            else:
                line = line.strip()
                if current_object is not None:
                    if line == "":
                        current_object = None
                        continue
                else:
                    if ":" in line:
                        class_name, suffix = line.split(":", 1)
                        if suffix != "":
                            raise ValueError(_NONEMPTY_CLASSNAME)
                        current_object = {"_classname": class_name}
                        objects.append(current_object)
                        continue

                if current_object is not None:
                    key, value = line.split(":", 1)
                    # TODO check if the key is supported for given class
                    if key.strip() == "":
                        # TODO msg
                        raise ValueError()
                    if value.strip() == "~":
                        multiline_value = ""
                        multiline_key = key
                    else:
                        value = parse_value(value, classname=class_name, key=key, context=context, autoparse=autocast)
                        current_object[key] = value
    except ValueError as ex:
        parsing_error = True
        error_msg = str(ex)

    if multiline_key is not None:
        current_object[multiline_key] = multiline_value
        warnings.warn(_MULTILINE_NOT_CLOSED_WARNING.format(multiline_key))

    if parsing_error:
        error_msc = "Parsing error. Incorrect syntax in line {}:\n{}\n{}".format(line_num, error_msg, line)
        raise ValueError(error_msc)

    return objects


def load(filename, context=None, autocast=True):
    """
    Parses the file with a given filename to a list of dictionaries.
    :param filename: Name of the file to parse.
    :param context: Context of parsing compliant with contexts found in 'framscript.xml' e.g. 'expdef file'.
    If context is left emtpy it will be inferred from the file's extension/
    :param autocast: If true numbers will be parsed automatically if possible.
        If false every field will be treated as a string.
        :return: A list of dictionaries representing Framsticks objects.
    """
    file = open(filename, encoding='UTF-8')
    if context is None:
        try:
            _, extension = filename.split(".")
            context = extension + " file"
            if context not in _contexts:
                context = None
                warnings.warn(_UNSUPPORTED_EXTENSION_WARNING.format(extension))
        except RuntimeError:
            warnings.warn(_NO_FILE_EXTENSION_WARNING)
            context = None
    s = file.read()
    file.close()
    return loads(s, context=context, autocast=autocast)
