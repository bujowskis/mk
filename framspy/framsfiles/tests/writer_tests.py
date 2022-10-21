#!/usr/bin/env python3
import glob
import json
import os
import unittest

from parameterized import parameterized

from ..reader import loads
from ..writer import from_collection


class SerializedTest(unittest.TestCase):
    def test_keyword_serialization(self):
        in_obj = [{'f': '@Serialized:'}]
        expected = 'f:@Serialized:"@Serialized:"\n'
        actual = from_collection(in_obj)
        self.assertEqual(actual, expected)

    def test_list_serialization(self):
        in_obj = [{'f': [16, None, 'abc']}]
        expected = 'f:@Serialized:[16,null,"abc"]\n'
        actual = from_collection(in_obj)
        self.assertEqual(actual, expected)

    def test_object_serialization(self):
        in_obj = [{'f': {'a': 123, 'b': [7, 8, 9]}}]
        expected = 'f:@Serialized:{"a":123,"b":[7,8,9]}\n'
        actual = from_collection(in_obj)
        self.assertEqual(actual, expected)

    def test_empty_nested_list_serialization(self):
        in_obj = [{'f': [[[]]]}]
        expected = 'f:@Serialized:[[[]]]\n'
        actual = from_collection(in_obj)
        self.assertEqual(actual, expected)

    def test_list_with_newlines_serialization(self):
        in_obj = [{'f': ['\n\n', '']}]
        expected = 'f:@Serialized:["\n\n",""]\n'
        actual = from_collection(in_obj)
        self.assertEqual(actual, expected)


class ClassnameTest(unittest.TestCase):
    def test_classname_detection(self):
        in_obj = [{'_classname': 'stats', 'a': 'b'}]
        expected = 'stats:\na:b\n'
        actual = from_collection(in_obj)
        self.assertEqual(actual, expected)

    def test_multiple_classname_detection(self):
        in_obj = [{'_classname': 'stats', 'a': 'b'}, {'_classname': 'org', 'c': 'd'}]
        expected = 'stats:\na:b\n\norg:\nc:d\n'
        actual = from_collection(in_obj)
        self.assertEqual(actual, expected)


class MultilineTest(unittest.TestCase):
    def test_tilde_wrap(self):
        in_obj = [{'ml': 'this is\na field\nwith multiple lines'}]
        expected = 'ml:~\nthis is\na field\nwith multiple lines~\n'
        actual = from_collection(in_obj)
        self.assertEqual(actual, expected)

    def test_tilde_escape(self):
        in_obj = [{'ml': 'this is\na field\nwith multiple lines\n~ is also here'}]
        expected = 'ml:~\nthis is\na field\nwith multiple lines\n\\~ is also here~\n'
        actual = from_collection(in_obj)
        self.assertEqual(actual, expected)


json_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "test_files", "outputs")
input_files = glob.glob(os.path.join(json_root, '*', '*'))


class WriterReaderTest(unittest.TestCase):
    @parameterized.expand(input_files)
    def test_write_then_read(self, filename):
        with self.subTest(i=filename):
            with open(filename, encoding='UTF-8') as file:
                json_content = json.load(file)
                after_write = from_collection(json_content)
                after_read = loads(after_write)
                self.assertEqual(len(json_content), len(after_read))
                for i, o in zip(json_content, after_read):
                    self.assertDictEqual(i, o)


if __name__ == '__main__':
    unittest.main()
