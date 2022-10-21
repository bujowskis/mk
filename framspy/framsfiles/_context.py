import os.path
from xml.etree import ElementTree

_FRAMSCRIPT_XML_PATH = os.path.join((os.path.dirname(__file__)), '', 'framscript.xml')


def _create_specs_from_xml():
    contexts = {'sim file', 'gen file'}
    specs = dict()
    for child in _get_root():
        context = _get_context(child)
        classname = _get_name(child)
        context_key = (context, classname)
        specs[context_key] = dict()
        contexts.add(context)

        for element in child:
            if _is_field(element):
                spec = dict()
                key = _get_name(element)
                dtype = _get_type(element)
                if dtype is None:
                    continue

                if 'min' in element.attrib:
                    spec['min'] = _get_min(element, dtype)
                if 'max' in element.attrib:
                    spec['max'] = _get_max(element, dtype)

                spec['dtype'] = dtype
                specs[context_key][key] = spec

    return specs, contexts


def _get_type(node):
    # possible types as of 1.03.2021: {'float', 'untyped', 'integer', 'string', 'text', 'void'}
    type_att = node.attrib['type']
    if type_att == 'string':
        return str
    if type_att == 'integer':
        return int
    if type_att == 'float':
        return float
    return None


def _get_root():
    return ElementTree.parse(_FRAMSCRIPT_XML_PATH).getroot()


def _get_context(node):
    return node.attrib['context']


def _get_name(node):
    return node.attrib['name']


def _get_min(node, dtype):
    return dtype(node.attrib['min'])


def _get_max(node, dtype):
    return dtype(node.attrib['max'])


def _is_field(node):
    return node.tag == 'element' and 'type' in node.attrib


_specs, _contexts = _create_specs_from_xml()
