# This Dockerfile is used to build an image in which I can easily test
# sqlshell on a Linux distribution.

# 3.12 on Debian bookworm
FROM python:bookworm

RUN apt-get update
RUN apt -y install git
RUN apt -y install python3
RUN apt -y install python3-pip
RUN apt -y install python3-venv
RUN apt -y install man
RUN apt -y install sudo
RUN apt -y install vim
RUN apt -y install zsh
RUN apt -y install wamerican-large
RUN useradd -ms /usr/bin/zsh bmc
USER bmc
WORKDIR /home/bmc
COPY inputrc /home/bmc/.inputrc
COPY editrc /home/bmc/.editrc
COPY zshrc /home/bmc/.zshrc
RUN mkdir -p /home/bmc/etc/findwords
RUN grep -E '^[a-z]+$' /usr/share/dict/words >/home/bmc/etc/findwords/dict.txt
RUN /usr/bin/python3 -m venv python
RUN sh -c ". ./python/bin/activate; pip install build"
RUN git clone https://github.com/bmc/py-findwords
RUN sh -c ". ./python/bin/activate; cd py-findwords; ./build.sh build"
RUN sh -c ". ./python/bin/activate; pip install py-findwords/dist/*.whl"
CMD /usr/bin/zsh -i
