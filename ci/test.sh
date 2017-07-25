#!/bin/sh
pep8 --ignore=E501 **/*.py && \
nosetests -s --nologcapture test
