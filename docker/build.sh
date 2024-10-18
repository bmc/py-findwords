#!/usr/bin/env bash
#
# Convenience script to build the Docker image. Uses image name
# bclapper/py-findwords

set -x
docker build -t bclapper/py-findwords .
