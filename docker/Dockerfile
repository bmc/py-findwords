# This Dockerfile is used to build an image in which I can easily test
# sqlshell on a Linux distribution.

# 3.13 on Debian bookworm
FROM python:3.13-bookworm

RUN ["apt-get", "update"]
RUN ["apt", "-y", "install", "git"]
RUN ["apt", "-y", "install", "python3"]
RUN ["apt", "-y", "install", "python3-pip"]
RUN ["apt", "-y", "install", "python3-venv"]
RUN ["apt", "-y", "install", "man"]
RUN ["apt", "-y", "install", "sudo"]
RUN ["apt", "-y", "install", "vim"]
RUN ["apt", "-y", "install", "zsh"]
RUN ["apt", "-y", "install", "wamerican-large"]
RUN ["apt", "-y", "install", "wget"]
RUN ["useradd", "-ms", "/usr/bin/zsh", "bmc"]
USER bmc
WORKDIR /home/bmc
COPY inputrc /home/bmc/.inputrc
COPY editrc /home/bmc/.editrc
COPY zshrc /home/bmc/.zshrc
RUN mkdir -p /home/bmc/etc/findwords
RUN ["sh", "-c", "grep -E '^[a-z]+$' /usr/share/dict/words >/home/bmc/etc/findwords/wamerican.txt"]
RUN ["wget", "-O", "/home/bmc/etc/findwords/dict.txt", "https://raw.githubusercontent.com/dwyl/english-words/master/words_alpha.txt"]
RUN ["/usr/local/bin/python3", "-m", "venv", "python"]
RUN ["sh", "-c", ". ./python/bin/activate; pip install build"]
# Check out the py-findwords repository, build it, and install it, so there's
# a locally installed version of the package.
RUN ["git", "clone", "https://github.com/bmc/py-findwords"]
RUN ["sh", "-c", ". ./python/bin/activate; cd py-findwords; ./build.sh build"]
RUN ["sh", "-c", ". ./python/bin/activate; pip install py-findwords/dist/*.whl"]
RUN ["sh", "-c", "sed 's/^min_length.*$/min_length = 3/' py-findwords/findwords.toml >/home/bmc/.findwords.toml"]
# Remove the locally checked out repo, since we'll mount the actual source
# when we run the container.
RUN ["rm", "-rf", "py-findwords"]

CMD ["/usr/bin/zsh", "-i"]
