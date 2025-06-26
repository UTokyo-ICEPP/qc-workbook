#!/bin/bash

VENV=$(dirname $(dirname $(which python)))
MINOR_VERSION=$(python -V | sed 's/Python \([0-9]*\.[0-9]*\)\.[0-9]*/\1/')

cd $VENV/lib/python${MINOR_VERSION}/site-packages

for dname in tests __pycache__ test-examples Tests _tests examples sample_data
do
  for dir in $(find . -name $dname -type d 2>/dev/null)
  do
    rm -r $dir
  done
done
for fname in LICENSE 'README*' dateutil-zoneinfo.tar.gz
do
  for file in $(find . -name $fname -type f 2>/dev/null)
  do
    rm $file
  done
done
for dir in $(find sphinx/locale -maxdepth 1 -mindepth 1 -type d 2>/dev/null)
do
  rm -r $dir
done
for dir in $(find . -type d -name locales 2>/dev/null)
do
  rm -r $dir
done
rm -rf babel/locale-data/ numpy/core/include/ pybind11/include/ pytz/zoneinfo 2>/dev/null
