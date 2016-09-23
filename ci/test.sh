#!/bin/sh
pep8 **/*.py && \
nosetests -s --nologcapture test
