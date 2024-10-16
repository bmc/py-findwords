#!/usr/bin/env bash
#
# Simple build script for py-sqlshell. Run as:
#
# ./build.sh target ...
#
# Valid targets: build, clean

usage() {
  # print usage to stderr and exit.
  echo "Usage: $0 target ..." >&2
  echo "       $0 -h" >&2
  echo "Valid targets: build, clean, docker >&2" >&2
  echo "Default: clean build"
  exit 1
}

run() {
  # Echo a shell command, run it, check the return code, and print an
  # error if the command fails. Returns 1 if the command fails, 0 otherwise.
  echo "+ $1"
  eval $1
  rc=$?
  if [ $rc != 0 ]
  then
    echo "--- Failed: $rc" >&2
    return 1
  fi
}

while getopts ":h" opt; do
  case $opt in
    h)
      usage
      ;;
    *)
      echo "Invalid option: -$OPTARG" >&2
      usage
      ;;
  esac
done
shift $((OPTIND -1))

case $# in
  0)
    targets="clean build"
    ;;
  1)
    targets="$1"
    ;;
  *)
    targets="$*"
    ;;
esac

# Validate targets
for t in $targets
do
  case $t in
    build|clean)
      ;;
    *)
      usage
      ;;
  esac
done

# Run targets
for t in $targets
do
  case $t in
    clean)
      run "rm -rf *.egg-info" || exit 1
      run "rm -rf dist" || exit 1
      run "rm -rf __pycache__" || exit 1
      run "rm -rf findwords/__pycache__" || exit 1
      ;;

    build)
      run "python -m build" || exit 1
      ;;
  esac
done
