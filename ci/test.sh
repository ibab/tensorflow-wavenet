#!/usr/bin/env bash

pep8 **/*.py && \
nosetests -s --nologcapture test
