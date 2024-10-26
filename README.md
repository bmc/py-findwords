# py-findwords

Written in Python, this program is a fast word game anagram finder, suitable
for use with Scrabble®, Words with Friends, and similar games. It also has an
interactive mode that supports history and command-line editing.

## Installation

This tool is _not_ in PyPI, and it likely never will be. You can install
it easily enough from the source code.

First, check out a copy of this repository. Then, you'll use the Python
[build](https://build.pypa.io/en/stable/index.html) tool to build the
`findwords` package, which you can then use `pip` to install. Example
below. Change the version number as appropriate.

```shell
$ git clone https://github.com/bmc/py-findwords.git
$ pip install build
$ cd py-findwords
$ ./build.sh clean build
$ pip install dist/findwords-1.0.0-py3-none-any.whl
```

It'll install a `findwords` command in your current Python environment.
(That environment really should be a virtual environment.)

## The dictionary

`findwords` requires a dictionary of words, a file containing one word
per line. **It does not come bundled with one.** You have to supply the
dictionary. Possible dictionaries include:

* The [Collins Scrabble Words](https://en.wikipedia.org/wiki/Collins_Scrabble_Words)
  list (formerly called "SOWPODS"). With some diligent searching, you might
  be able to find one online. At last updating of this file, there's a slightly
  older version [here](https://github.com/MReeveCO/sowpods).
* The [NASPA Word list](https://en.wikipedia.org/wiki/NASPA_Word_List),
  assuming you can find a copy in the right format.
* Something like `/usr/dict/words` (or equivalent) on Unix-like systems
  will also do (e.g., `/usr/share/dict/words` on MacOS).
* `words_alpha.txt` from
  [github.com/dwyl/english-words](https://github.com/dwyl/english-words) is
  also a reasonable choice.


`findwords` loads the dictionary into a lookup trie when it starts up.

### Specifying the dictionary

You can specify which dictionary file to load either via the `-d` (or,
`--dictionary`) command line option or via the `FINDWORDS_DICTIONARY`
environment variable. If you don't set either one of those, `findwords`
expects the find the dictionary in `$HOME/etc/findwords/dict.txt`. If it
can't find a dictionary, it aborts.

### Limitations

Currently, `findwords` only supports ASCII letters. When loading the
dictionary and when processing user input, it rejects any word that doesn't
consist entirely of ASCII letters. That includes accented letters, apostrophes,
hyphens, and the like.

## Running `findwords`

Once you've installed it, just type `findwords` in a terminal window.
Use `findwords -h` (or `findwords --help`) to get usage information. You
can specify one or more words (strings) on the command line, and it'll process
those and exit. If you omit words from the command, it'll go into interactive
mode, where you can enter as many words as you want.

Interactive mode uses the `readline` library, which means it keeps a history
of all previously entered strings and commands, up to a maximum of 10,000
entries. The history is stored in file `$HOME/.findwords-history`.

You can capture your desired defaults for most of the command-line options
via a configuration file, `$HOME/.findwords.toml`. See the sample
`findwords.toml` configuration file in at the top of this repo.

## Maintenance Warning

I built this tool for my personal use. It replaces an older version I'd written
in Scala. (This Python version has less code and starts up faster.)

If you find it useful, as a tool, as a reasonable enough example of code
that uses a [trie](https://en.wikipedia.org/wiki/Trie), or even as an example
of how to build a `readline`-based program in Python, that's great. But this
isn't intended to be commercial-grade software, and I'm not aggressively
maintaining it. Don't expect me to jump on feature requests or bug reports.

**Special note for Windows users**: You're on your own. Sorry. I haven't
tested `findwords` on Windows, and I likely won't. I dislike doing development
on Windows. `findwords` runs fine for me on Linux and MacOS. If it somehow
works for you on Windows, that's terrific (but unlikely). In any case, I'm
unwilling to spend any time getting it to work there. (You can always just
build the Docker image and run it under Docker on Windows. See the `docker`
subdirectory.)

## License

`findwords` is copyright © 2024 Brian M. Clapper and is released under the
[Apache Software License](https://apache.org/licenses/LICENSE-2.0), version
2.0. A text copy of the license is available in this repository.
