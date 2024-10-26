#!/usr/bin/env bash
#
# Convenience script to build the Docker image. Uses image name
# bclapper/py-findwords

function usage {
  echo "Usage: $0 [--no-cache]" >&2
  exit 1
}

case $# in
  0)
    opts=""
    ;;
  1)
    if [ "$1" = "--no-cache" ]
    then
      opts="--no-cache"
    else
      usage
    fi
    ;;
  *)
    usage
    ;;
esac

set -x
docker build $opts -t bclapper/py-findwords .
