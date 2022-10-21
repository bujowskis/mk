pushd ..\..
echo "Testing framswriter..."
python -m unittest framsfiles/tests/writer_tests.py
echo "Testing framsreader..."
python -m unittest framsfiles/tests/reader_tests.py
popd