#!/bin/bash

src=/usr/local/lib/python3.8/dist-packages
mkdir -p /packages/$src
cp -r $src/* /packages/$src
cd /packages/$src

for dname in tests test __pycache__ test-examples datasets Tests _tests examples sample_data
do
  find . -name $dname -type d | xargs rm -r
done
for fname in LICENSE 'README*' dateutil-zoneinfo.tar.gz
do
  find . -name $fname -type f | xargs rm -r
done
find sphinx/locale -maxdepth 1 -mindepth 1 -type d | xargs rm -r
find . -type d -name locales | xargs rm -r
rm -r babel/locale-data/
rm -r numpy/core/include/
rm -r pybind11/include/
rm -r pytz/zoneinfo

src=/usr/lib/python3/dist-packages
mkdir -p /packages/$src
cp -r $src/pkg_resources /packages/$src
cp -r $src/setuptools* /packages/$src

src=/usr/lib/python3.8
mkdir -p /packages/$src
cp -r $src/distutils /packages/$src

cd /packages
tar czf ../packages.tar.gz *
