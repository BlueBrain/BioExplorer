#!/bin/sh

rm -rf build
rm -rf source/api
make html
cp metadata.md build/html
