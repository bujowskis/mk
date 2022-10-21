#!/usr/bin/env python3
import unittest
from parameterized import parameterized
import os
import json
import re
import glob

from ..reader import loads, load, parse_value

test_files_root = "test_files"
input_files_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), test_files_root, "inputs")
output_files_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), test_files_root, "outputs")
input_files = glob.glob(os.path.join(input_files_root, '*', '*'))

parse_value_testcases = [
    ('1', 1),
    ('0', 0),
    ('a', 'a'),
    # TODO tilde should also be escaped in single-line values (now it is only escaped in multi-line values)
    ('generic string ~ with tylda', 'generic string ~ with tylda'),
    ('123', 123),
    ('0x123', 0x123),
    ('0X123', 0x123),
    ('+0X10', 16),
    ('-0x125', -0x125),
    ('-0xaa', -0xaa),
    ('0xaa', 0xaa),
    ('0xAA', 0xaa),
    ('-12.3', -12.3),
    ('12.3e+05', 12.3e+05),
    ('@Serialized:null', None),
    ('@Serialized: null', None),
    ('@Serialized:"@Serialized:"', '@Serialized:'),
    ('@Serialized:"\\""', '"'),
    ('@Serialized:"\\t"', '\t'),
    ('@Serialized:"\\n"', '\n'),
    ('@Serialized:1', 1),
    ('@Serialized:2.1', 2.1),
    ('@Serialized:+2.1', 2.1),
    ('@Serialized:-2.1', -2.1),
    ('@Serialized: +0X10  ', 16),
    ('@Serialized: 0X10  ', 16),
    ('@Serialized: 0XAA  ', 0xaa),
    ('@Serialized: 0Xaa  ', 0xaa),
    ('@Serialized:[]', []),
    ('@Serialized:[1,2,3]', [1, 2, 3]),
    ('@Serialized: [ +0X10 ] ', [16]),
    ('@Serialized:[ +0X10, null, "abc"] ', [16, None, "abc"]),
    ('@Serialized:[[[]]]', [[[]]]),
    ('@Serialized:{}', {}),
    ('@Serialized:{"a":123 }', {"a": 123}),
    ('@Serialized:{"a":123,"b":2 }', {"a": 123, "b": 2}),
    ('@Serialized:{"a":123,"b":[1,2,3] }', {"a": 123, "b": [1, 2, 3]}),
    ('@Serialized:XYZ[0,1,2]', (0, 1, 2)),
    ('@Serialized:Population<0x85f53a8>', 'Population<0x85f53a8>'),
    ('@Serialized:CrazyObject[{},{},[[]]]', 'CrazyObject[{},{},[[]]]'),
    ('@Serialized:[co[{},{},[[]]],[]]', ['co[{},{},[[]]]', []]),
    ('@Serialized:[co[{},{},[[]]],[1,2,3]]', ['co[{},{},[[]]]', [1, 2, 3]]),
    ('@Serialized:[[1,2, 3],  co[{},{},[[]]]]', [[1, 2, 3], 'co[{},{},[[]]]']),
    # TODO maybe raise if there is a space?
    ('@Serialized:Population <0x85f53a8>', 'Population <0x85f53a8>'),

]
context_parse_value_testcases = [
    ({'value': '1', 'classname': 'expdef', 'key': 'name', 'context': 'expdef file'}, '1'),
    ({'value': '234.5', 'classname': 'Creature', 'key': 'energ0', 'context': 'Global Context'}, 234.5)
]
parse_value_exception_testcases = [
    '@Serialized:   '
]

loads_testcases = [
    ('class:\nmlprop:~\nbla bla bla\n~\n', [{"_classname": "class", "mlprop": "bla bla bla\n"}]),
    ('class:\nmlprop:~\n\\~\n~\n', [{"_classname": "class", "mlprop": "~\n"}])
]
loads_exception_testcases = [
    'class:\nmlprop:~\n\\~\n~\nasdasd',
    'class:\nmlprop:~\n~\n~\n',
]


# TODO make more atomic tests, maybe
class ReferenceTest(unittest.TestCase):
    def test0(self):
        str_in = '@Serialized:[^0] '
        result = parse_value(str_in)
        self.assertTrue(isinstance(result, list))
        self.assertEqual(len(result), 1)
        self.assertTrue(result is result[0])

    def test1(self):
        str_in = '@Serialized:[44,[^1]]'
        result = parse_value(str_in)
        self.assertTrue(isinstance(result, list))
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0], 44)
        self.assertTrue(result[1] is result[1][0])

    def test2(self):
        str_in = '@Serialized:[[100],["abc"],[300,^2]]'
        result = parse_value(str_in)
        self.assertTrue(isinstance(result, list))
        self.assertEqual(len(result), 3)
        self.assertEqual(result[0], [100])
        self.assertEqual(result[1], ["abc"])
        self.assertEqual(result[2][0], 300)
        self.assertTrue(result[2][1] is result[1])

    def test3(self):
        str_in = '@Serialized:[[123,[]],["x",^0],^2]'
        result = parse_value(str_in)
        self.assertTrue(isinstance(result, list))
        self.assertEqual(len(result), 3)
        self.assertEqual(result[0], [123, []])
        self.assertTrue(isinstance(result[1], list))
        self.assertEqual(result[1][0], "x")
        self.assertTrue(result[2] is result[0][1])
        self.assertTrue(result[1][1] is result)

    def test4(self):
        str_in = '@Serialized:{"a":[33,44],"b":^1,"c":[33,44]}'
        result = parse_value(str_in)
        self.assertTrue(isinstance(result, dict))
        self.assertEqual(len(result), 3)
        self.assertListEqual(sorted(result.keys()), ["a", "b", "c"])

        self.assertEqual(result["a"], [33, 44])
        self.assertEqual(result["c"], [33, 44])

        self.assertFalse(result["c"] is result["a"])
        self.assertTrue(result["b"], result["a"])

    def test5(self):
        str_in = '@Serialized:[null, null, [1, 2], null, ^ 1]'
        result = parse_value(str_in)
        self.assertTrue(isinstance(result, list))
        self.assertEqual(len(result), 5)
        self.assertListEqual(result[0:4], [None, None, [1, 2], None])
        self.assertTrue(result[2] is result[4])


class ParseValueTest(unittest.TestCase):
    @parameterized.expand(parse_value_testcases)
    def test_correct_parsing(self, input_val, output):
        self.assertEqual(output, parse_value(input_val))

    @parameterized.expand(context_parse_value_testcases)
    def test_correct_context_parsing(self, input_val, output):
        self.assertEqual(output, parse_value(**input_val))

    @parameterized.expand(parse_value_exception_testcases)
    def test_parsing_exceptions(self, input_val):
        self.assertRaises(ValueError, parse_value, input_val)


class LoadsTest(unittest.TestCase):
    @parameterized.expand(loads_testcases)
    def test_correct_loads(self, input_val, output):
        self.assertEqual(output, loads(input_val))

    @parameterized.expand(loads_exception_testcases)
    def test_load_exceptions(self, input_val):
        self.assertRaises(ValueError, loads, input_val)


class LoadTest(unittest.TestCase):
    @parameterized.expand(input_files)
    def test_correct_load(self, filename):
        json_output_path = output_files_root + filename[len(input_files_root):] + ".json"
        with self.subTest(i=filename):
            result = load(filename)
            with open(json_output_path, encoding='UTF-8') as json_file:
                correct = json.load(json_file)
                self.assertEqual(len(result), len(correct))
                for r, c in zip(result, correct):
                    self.assertDictEqual(r, c)


if __name__ == '__main__':
    unittest.main()
