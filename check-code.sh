#!/usr/bin/env bash
#
# Run Python checkers and formatters.

case $# in
  0)
    args="."
    ;;
  *)
    args="$@"
    ;;
esac

die() {
  echo "$0: $1" >&2
  exit 1
}

for i in $args
do
  # pycheck is a personal tool in a private repo. Feel
  # free to comment this out.
  if [ "$i" = "." ]
  then
    echo "Checking types in findwords"
    pyright findwords || die "pyright failed"
    echo "Linting findwords"
    pylint findwords || die "pylint failed"
    echo "pycheck -cf " findwords/*.py
    pycheck -cf findwords/*.py || die "pycheck failed"
  else
    echo "Checking types in $i"
    pyright $i || die "pyright failed"
    echo "Linting $i"
    pylint $i || die "pylint failed"
    echo "pycheck -cf $i"
    pycheck -cf $i || die "pycheck failed"
  fi

  #echo "Sorting imports in $i"
  #isort $i

  #echo "Formatting $i with black"
  #black $i
done
