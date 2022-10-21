#!/bin/bash
test_dir=$(realpath $0 | xargs dirname)/../..
echo "Testing framswriter..."
cd $test_dir && python -m unittest framsfiles/tests/writer_tests.py
echo "Testing framsreader..."
cd $test_dir && python -m unittest framsfiles/tests/reader_tests.py