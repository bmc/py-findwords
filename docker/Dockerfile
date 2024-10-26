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
RUN apt -y install wget
RUN useradd -ms /usr/bin/zsh bmc
USER bmc
WORKDIR /home/bmc
COPY inputrc /home/bmc/.inputrc
COPY editrc /home/bmc/.editrc
COPY zshrc /home/bmc/.zshrc
RUN mkdir -p /home/bmc/etc/findwords
RUN grep -E '^[a-z]+$' /usr/share/dict/words >/home/bmc/etc/findwords/wamerican.txt
RUN wget -O /home/bmc/etc/findwords/dict.txt https://raw.githubusercontent.com/dwyl/english-words/master/words_alpha.txt
RUN /usr/bin/python3 -m venv python
RUN sh -c ". ./python/bin/activate; pip install build"
RUN git clone https://github.com/bmc/py-findwords
RUN sh -c ". ./python/bin/activate; cd py-findwords; ./build.sh build"
RUN sh -c ". ./python/bin/activate; pip install py-findwords/dist/*.whl"
RUN sed 's/^min_length.*$/min_length = 3/' py-findwords/findwords.toml >/home/bmc/.findwords.toml
# Remove the locally checked out version, since we'll mount the actual source
# when we run the container.
RUN rm -rf py-findwords

CMD /usr/bin/zsh -i
