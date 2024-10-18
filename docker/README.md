This directory contains files to build a Docker image that allows me to
test `findwords` in a Linux-like environment without leaving my current
development environment (which is MacOS). It's only used during development
and testing.

Run `./build.sh` to build the Docker image locally.

Run, from the main `py-findwords` directory, as:

```shell
$ docker run -ti -v $PWD:/home/bmc/py-findwords bclapper/py-findwords
```
