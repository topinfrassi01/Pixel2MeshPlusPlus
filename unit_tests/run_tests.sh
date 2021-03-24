#!/bin/bash

if [ "$1" != 'norebuild' ]
then
echo 'Building sources and virtual env...'
rm -r ../source/build >/dev/null
rm -r ../source/dist >/dev/null
rm -r ../source/sourcemodules.egg-info >/dev/null
python -m build ../source >/dev/null
rm -r modules-dist/*
mv ../source/dist/* modules-dist
python -m venv --system-site-packages tests >/dev/null
fi

source tests/bin/activate

if [ "$1" != 'norebuild' ]
then
echo 'Installing source modules...'
pip install pytest >/dev/null
pip install --ignore-installed modules-dist/sourcemodules-0.0.1-py3-none-any.whl >/dev/null
fi

for testfile in test_*; do
echo 'Running test file : ' $testfile
echo '-------------------------------'
pytest $testfile --verbose --disable-warnings
echo '-------------------------------'
done

if [ "$1" != 'norebuild' ]
then
echo 'Cleaning build...'
rm -r ../source/build
rm -r ../source/dist
rm -r ../source/sourcemodules.egg-info
fi
