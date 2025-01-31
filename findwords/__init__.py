"""
Take a string of letters and find all words that can be made from them.
See the README.md file for more information.
"""
import atexit
import dataclasses
from dataclasses import dataclass, field
from enum import StrEnum
import itertools
import os
from pathlib import Path
import random
import re
import readline
import string
import textwrap
import tomllib
from time import time
from typing import Self, Callable, Sequence, Tuple, Any

import art
import click
from termcolor import colored


NAME = "findwords"
VERSION = "1.1.2"
CLICK_CONTEXT_SETTINGS = {"help_option_names": ["-h", "--help"]}
HISTORY_LENGTH = 10_000
# Note that Python's readline library can be based on GNU Readline
# or the BSD Editline library, and it's not selectable. It's whatever
# has been compiled in. They use different initialization files, so we'll
# load whichever one is appropriate.
EDITLINE_BINDINGS_FILE = Path("~/.editrc").expanduser()
READLINE_BINDINGS_FILE = Path("~/.inputrc").expanduser()
DEFAULT_SCREEN_WIDTH = 79
DEFAULT_HISTORY_FILE = Path("~/.findwords-history").expanduser()
DEFAULT_CONFIG_FILE = Path("~/.findwords.toml").expanduser()
DEFAULT_DICTIONARY = Path("~/etc/findwords/dict.txt").expanduser()
DEFAULT_MIN_LENGTH = 2
PROMPT = colored("findwords> ", "cyan", attrs=["bold"])
DEFAULT_SCREEN_WIDTH: int = 79
INTERNAL_COMMAND_PREFIX = "."
ALL_DIGITS = re.compile(r"^\d+$")
ART_FONTS = (
    "big",
    "chunky",
    "cybermedium",
    "smslant",
    "speed",
    "standard",
    "tarty2",
    "tarty3",
)
BANNER_COLORS = (
    "cyan",
    "red",
    "black",
    "blue",
    "green",
)


@dataclass(frozen=True)
class Params:
    """
    Class to hold command line and configuration parameters.
    """
    dictionary: Path
    history_path: Path
    min_length: int
    verbose: bool


@dataclass
class TrieNode:
    """
    This class represents a node in the dictionary trie. The dictionary
    trie is a normal, not a compressed, trie such that each node represents
    a letter, and the words that can be made from that letter and all preceding
    nodes are stored in the node.

    To load the dictionary, we process each word as follows: The word is first
    sorted so that its letters are in alphabetical order. Then, each letter
    becomes a node in the trie. If there's a word that can be made with the
    letters up to that point in the trie, then it's added to the list of words
    at that node. Thus, traversing the path "a" -> "a" -> "b", you'll find
    "baa" in the list of words at node "b". At any given node, it's
    theoretically possible to have all 26 letters as children. The root node
    has no letter prefix at all; it just has children.

    To find all matches, you basically start at the root and recursively
    check each child. At each child, you save the words at that child, then
    remove the first instance of the child's letter and recursively check its
    children with the new string. See find_matches(), below.

    Example: Find all matches for "alphabet"

    1. Sort "alphabet", so that you get "aabehlpt"
    2. Start with the root node. Remove all children whose letters aren't
       in "aabehlpt"
    3. Get the child for the first letter ("a"). Save its words, if any.
    4. Then, remove the first "a", yielding "abehlpt"
    5. Get all the children of node "a", and recursively match against
       "abehlpt", etc.
    6. Then, go back and do the same thing with the "b", then the "e", etc.

    See find_matches() for algorithmic details.
    """

    letter: str | None = field(default=None)
    children: dict[str, Self] = field(default_factory=dict)
    words: set[str] = field(default_factory=set)

    def find_matches(self: Self, letters: str, min_length: int) -> list[str]:
        """
        Given a group of letters, find all words in the dictionary that can
        be made from them. This method must be called on the root node, or
        it will not be able to search the entire dictionary. It does not,
        however, check that it's being called on the root node.

        :param letters: the letters to check
        :param root: the root of the loaded dictionary trie

        :return: a possibly empty list of words that match
        """

        def search(letters: str, nodes: list[TrieNode]) -> list[str]:
            """
            Recursive function to search for matches.

            :param letters: the letters to match
            :param nodes: the (child) nodes to search

            :return: a list of words that match
            """
            def remove_first(letter: str, letters: str) -> str:
                """
                Remove the first letter matching "letter" from the supplied
                string of letters, if it exists. If it doesn't exist, just
                return "letters".
                """
                if (i := letters.index(letter)) != -1:
                    return letters[:i] + letters[i + 1 :]

                # Letter not found. Just return the string.
                return letters

            # Filter the list of nodes so that we only consider the ones
            # that matter.
            nodes = [
                n for n in nodes if n.letter is not None and n.letter in letters
            ]

            matches = []
            # Now, recursively check each one.
            for n in nodes:
                # This node's children represent possible paths. Add any words
                # in this node, since it's a match.
                matches.extend(list(n.words))

                # Remove the first instance of n.letter from the letters, and
                # recursively check the child nodes of this node.
                assert n.letter is not None
                sub_letters = remove_first(n.letter, letters)
                child_list = list(n.children.values())
                matches.extend(search(sub_letters, child_list))

            return matches

        sorted_letters = "".join(sorted(letters.lower()))
        all_matches = search(sorted_letters, list(self.children.values()))
        return [m for m in all_matches if len(m) >= min_length]


class InternalCommand(StrEnum):
    """
    Commands that can be issued interactively.
    """

    EXIT = f"{INTERNAL_COMMAND_PREFIX}exit"
    HELP = f"{INTERNAL_COMMAND_PREFIX}help"
    HISTORY = f"{INTERNAL_COMMAND_PREFIX}history"
    RERUN = "!"  # special case: this is a prefix


# This is a series of (command, explanation) tuples, used to generate help
# output.
HELP: Sequence[Tuple[str, str]] = (
    (InternalCommand.EXIT.value, f"Quit {NAME}. You can also use Ctrl-D."),
    (InternalCommand.HELP.value, "This output."),
    (
        f"{InternalCommand.HISTORY.value} [<n>]",
        (
          "Show the command and word history. If <n> is specified, show only "
          "the last <n> history items."
        )
    ),
    (
        f"{InternalCommand.RERUN.value}<n>",
        (
            "Rerun command <n> from the history. "
            f"{InternalCommand.RERUN.value}{InternalCommand.RERUN.value} reruns the "
            "most recent command."
        ),
    ),
)

HELP_EPILOG = (
    f'Anything not starting with "{INTERNAL_COMMAND_PREFIX}" or '
    f'"{InternalCommand.RERUN.value}" is interpreted as letters to be matched '
    "against dictionary words."
)


# Will be changed to something else if -q is specified.
verbose_msg: Callable[[str], None] = print


match os.environ.get("COLUMNS"):
    case None:
        SCREEN_WIDTH = DEFAULT_SCREEN_WIDTH
    case s_width:
        try:
            SCREEN_WIDTH = int(s_width)
        except ValueError:
            SCREEN_WIDTH = DEFAULT_SCREEN_WIDTH
            print(
                "The COLUMNS environment variable has an invalid value of "
                f'"{s_width}". Using screen width of {DEFAULT_SCREEN_WIDTH}.'
            )


def check_string(s: str, min_length: int) -> bool:
    """
    Determine whether a string to match is valid. Currently, only strings
    with ASCII letters are permitted, and the length must be at least as
    long as min_length.

    :param s: the string to check

    :raises ValueError: if the string is invalid
    """
    if len(s) < min_length:
        raise ValueError(
            f'"{s}" is shorter than the minimum match length of {min_length}'
        )

    for letter in s.lower():
        if letter not in string.ascii_lowercase:
            raise ValueError(
                f'"{s}" contains non-ASCII characters or non-letters.'
            )

    return True


def time_op(func: Callable[..., Any], *args: Any, **kw: Any) -> Tuple[int, Any]:
    """
    Time a function call, in milliseconds.

    :param func: the function to call
    :param args: positional arguments to pass to the function
    :param kw: keyword arguments to pass to the function

    :return: a tuple of the elapsed milliseconds and the result of the call
    """
    start = time()
    result = func(*args, **kw)
    elapsed = int((time() - start) * 1000)
    return elapsed, result


def add_word(word: str, root: TrieNode) -> None:
    """
    Add a word from the dictionary into the dictionary trie.

    :param word: the word to add
    :param root: the root of the trie
    """
    node = root

    # Build node tree for the word. The final node gets the word attached
    # to it.
    for letter in sorted(word.lower()):
        if (subnode := node.children.get(letter)) is None:
            subnode = TrieNode(letter)
            node.children[letter] = subnode

        node = subnode

    node.words.add(word)


def load_dictionary(dict_path: Path, min_length: int) -> TrieNode:
    """
    Load the dictionary into a trie.

    :param dict_path: path to the dictionary file to load
    :param min_length: minimum length of words to load

    :return: the root of the dictionary trie
    """
    root = TrieNode()

    def do_load() -> int:
        """
        Load the dictionary into the trie. This is the actual workhorse
        function, wrapped by time_op() in the parent function.
        """
        total = 0
        unique_words: set[str] = set()
        with open(dict_path, encoding="utf-8") as f:
            verbose_msg(f'Loading dictionary "{dict_path}".')
            for line in f.readlines():
                word = line.strip()
                if len(word) < min_length:
                    continue

                try:
                    check_string(word, 1)
                except ValueError as e:
                    verbose_msg(f'*** Skipping word "{word}": {e}')
                    continue

                if word in unique_words:
                    continue

                unique_words.add(word)
                add_word(word, root)
                total += 1

        return total

    elapsed, total = time_op(do_load)
    verbose_msg(f"Loaded {total:,} words in {elapsed / 1000:.2f} second(s).")

    return root


def init_readline_history(history_path: Path) -> None:
    """
    Load the local readline history file.

    :param history_path: Path of the history file. It doesn't have to exist.
    """
    if history_path.exists():
        verbose_msg(f'Loading history from "{history_path}".')
        readline.read_history_file(str(history_path))
        # default history len is -1 (infinite), which may grow unruly

    readline.set_history_length(HISTORY_LENGTH)
    atexit.register(readline.write_history_file, str(history_path))


def init_readline_bindings() -> None:
    """
    Initialize readline bindings. Completion isn't currently necessary,
    since there are no commands to complete.
    """
    if (readline.__doc__ is not None) and ("libedit" in readline.__doc__):
        init_file = EDITLINE_BINDINGS_FILE
        verbose_msg("Using editline (libedit).")
        completion_binding = "bind '^I' rl_complete"
    else:
        init_file = READLINE_BINDINGS_FILE
        verbose_msg("Using GNU readline.")
        completion_binding = "Control-I: rl_complete"

    if init_file.exists():
        verbose_msg(f'Loading readline bindings from "{init_file}".')
        readline.read_init_file(init_file)

    readline.parse_and_bind(completion_binding)


def init_readline_completion() -> None:
    """
    Initialize readline completion for "." (internal) commands.
    """

    def command_completer(text: str, state: int) -> str | None:
        """
        Completes internal commands starting with "."
        """
        commands = [cmd.value for cmd in InternalCommand]
        full_line = readline.get_line_buffer()

        # Get the first token. If it matches a complete command, handle it
        # differently than if it's a partial match.
        tokens = full_line.lstrip().split()
        match tokens:
            case []:
                options = commands
            case [s, *_] if s in commands:
                # Already completed command. There's nothing else to complete.
                options = []
            case [s]:
                options = [c for c in commands if c.startswith(text)]
            case _:
                options = []

        return options[state] if state < len(options) else None

    readline.set_completer(command_completer)


def init_readline(history_path: Path) -> None:
    """
    Initializes all necessary aspects of the readline library.
    """
    init_readline_history(history_path)
    init_readline_bindings()
    init_readline_completion()


def show_help() -> None:
    """
    Interactive mode only: Show command help.
    """
    prefix_width = 0
    for cmd, _ in HELP:
        prefix_width = max(prefix_width, len(cmd))

    # How much room do we have left for text? Allow for separating " - ".

    separator = " - "
    text_width = SCREEN_WIDTH - len(separator) - prefix_width
    if text_width < 0:
        # Screw it. Just pick some value.
        text_width = DEFAULT_SCREEN_WIDTH // 2

    for cmd, text in HELP:
        cmd_str = cmd
        padded_prefix = colored(
            cmd_str.ljust(prefix_width), "red", attrs=["bold"]
        )
        text_lines = textwrap.wrap(text, width=text_width)
        print(f"{padded_prefix}{separator}{text_lines[0]}")
        for text_line in text_lines[1:]:
            padding = " " * (prefix_width + len(separator))
            print(f"{padding}{text_line}")

    print("")
    wrapped = textwrap.fill(HELP_EPILOG, width=SCREEN_WIDTH)
    print(wrapped)


def get_full_history() -> list[Tuple[int, str]]:
    """
    Return the entire readline() history as a list to (number, string) pairs.
    The number is the history item ID number, and it will be unique and in
    ascending order, starting at 1.
    """
    history_length = readline.get_current_history_length()
    # History indexes are 1-based, not 0-based
    return [
        (i, readline.get_history_item(i)) for i in range(1, history_length)
    ]


def show_history(total: int = 0) -> None:
    """
    Interactive mode only: Display the history.

    :param total: Limit to number of entries to show, or 0 for all. NOT
                  CURRENTLY USED. Hook for future enhancement.
    """

    def format_history_item(line: str, index: int) -> str:
        """
        Format a single history line.

        :param line: The history line
        :param index: The index (number) of the history line
        """
        return f"{index:5d}) {line}"

    history_items = get_full_history()
    match total:
        case n if n <= 0:
            pass
        case n if n > 0:
            history_items = history_items[-n:]

    for i, line in history_items:
        print(format_history_item(line, i))


def multiple_matches_header(s: str) -> str:
    """
    Return a suitable header for a string being matched. Useful when
    emitting output for multiple strings.

    :param s: the string
    :return: the header
    """
    sep = "*" * len(s)
    return f"{sep}\n{s}\n{sep}"


def show_matches(matches: list[str]) -> None:
    """
    Display all matches for a set of letters. The matches are grouped by
    word length and then sorted within each group, so that all matches of
    the same length are printed together. Each group is separated by a blank
    line, for readability.

    :param matches: the list of words that match the original input string
    """
    if len(matches) == 0:
        print("*** No matches.")
        return

    # Group them by word length. Note that itertools.groupby() requires that
    # the input be sorted by the same function that will group them.
    sorted_matches = sorted(matches, key=len)
    grouped = itertools.groupby(sorted_matches, key=len)

    # groupby() returns a (key, group_list) pair. The key is the length, which
    # we can ignore here.
    for _, group in grouped:
        for word in sorted(group):
            print(word)

        print()


def find_matches_for_inputs(strings: list[str],
                            trie: TrieNode,
                            min_length: int) -> None:
    """
    Break the line up into multiple words, to allow more than one
    word per line; then, for each of the words, find and display the
    matches.

    :param strings: the list of strings to match
    :param trie: the loaded dictionary trie
    :param min_length: minimum length of words to match
    """
    use_prefix = len(strings) > 1
    for s in strings:
        try:
            check_string(s, min_length)
        except ValueError as e:
            print(f'Ignoring "{s}": {e}')
            continue

        if use_prefix:
            print()
            print(multiple_matches_header(s))
            print()

        show_matches(trie.find_matches(s, min_length))


def handle_history_rerun(line: str, trie: TrieNode, min_length: int) -> bool:
    """
    Parse and process the "history rerun" command, returning
    True if the user wants to exit (i.e., re-runs an exit command)
    and False otherwise.

    :param line: the line of input representing the command
    :param trie: the loaded dictionary trie
    :param min_length: minimum length of words to match
    """
    assert line[0] == InternalCommand.RERUN.value
    history = get_full_history()

    def rerun(command: str) -> bool:
        """
        Display and rerun a command from the history.
        """
        print(f"{PROMPT}{command}")
        return handle_command(command, trie, min_length)

    exit_requested = False

    match line[1:]:
        case InternalCommand.RERUN.value:
            # rerun last command
            if len(history) == 0:
                print("History is empty.")
            else:
                exit_requested = rerun(history[-1][1])

        case s if ALL_DIGITS.search(s) is not None:
            # rerun command n
            n = int(s)
            history_dict = dict(history)
            cmd = history_dict.get(n)
            if cmd is None:
                print(f"There's no history item #{n}.")
            else:
                exit_requested = rerun(cmd)

        case _:
            print(
                f"{InternalCommand.RERUN.value} must either be followed "
                f"by a number or by {InternalCommand.RERUN.value}"
            )

    return exit_requested


def handle_command(line: str, trie: TrieNode, min_length: int) -> bool:
    """
    Handle command input. Return True if user wants to exit,
    False otherwise.

    :param line: the line of input representing the command
    :param trie: the loaded dictionary trie
    :param min_length: minimum length of words to match
    """
    if len(line) == 0:
        return False

    exit_requested = False

    line = line.strip()
    tokens = line.split()
    match tokens:
        case []:
            pass
        case [InternalCommand.EXIT.value]:
            return True
        case [InternalCommand.EXIT.value, *_]:
            print(f"{InternalCommand.EXIT.value} takes no arguments.")
        case [InternalCommand.HELP.value]:
            show_help()
        case [InternalCommand.HELP.value, *_]:
            print(f"{InternalCommand.HELP.value} takes no arguments.")
        case [InternalCommand.HISTORY.value]:
            show_history()
        case [InternalCommand.HISTORY.value, n]:
            try:
                n = int(n)
                if n <= 0:
                    raise ValueError("Must be positive")
                show_history(n)
            except ValueError:
                print(f"{InternalCommand.HISTORY.value}: Invalid number.")
        case [InternalCommand.HISTORY.value, *_]:
            print(f"{InternalCommand.HISTORY.value}: Too many parameters.")
        case [s] if s[0] == InternalCommand.RERUN.value:
            # Special case: This is a prefix, followed by a history
            # item number.
            exit_requested = handle_history_rerun(line, trie, min_length)
        case [s, *_] if s[0] == INTERNAL_COMMAND_PREFIX:
            print(f'"{s}" is an unknown command.')
        case strings:
            find_matches_for_inputs(strings, trie, min_length)

    return exit_requested

def interactive_mode(
    trie: TrieNode, history_path: Path, min_length: int
) -> None:
    """
    No letters on command line, so prompt for successive words with
    readline. Continues prompting until Ctrl-D or InternalCommand.EXIT.

    :param trie: the loaded dictionary trie
    :param history_path: path to the history file to use
    :param min_length: minimum length of words to match
    """

    def print_banner() -> None:
        """
        Print the banner using a random art font and a random terminal
        foreground color. The generated figlet has some blank lines (vertical
        padding at the end, which we will remove.
        """
        banner: str = art.text2art(NAME, rng.choice(ART_FONTS)) # type: ignore
        banner_lines: list[str] = banner.split("\n")
        color: str = rng.choice(BANNER_COLORS)
        while (len(banner_lines) > 0) and (banner_lines[-1].strip() == ""):
            banner_lines.pop()

        print(colored("\n".join(banner_lines), color))
        print()

    init_readline(history_path)

    rng = random.SystemRandom()
    print_banner()
    print(f"Version {VERSION}")
    print("Enter one or more strings, separated by white space.")
    print(f'Type Ctrl-D or "{InternalCommand.EXIT.value}" to exit.')
    print(f'Type "{InternalCommand.HELP.value}" for help on commands.')
    print()

    while True:
        try:
            # input() automatically uses the readline library, if it's
            # been loaded.
            exit_requested = handle_command(input(PROMPT), trie, min_length)
            if exit_requested:
                break

        except EOFError:
            # Ctrl-D to input()
            print()
            break


def once_and_done(
    trie: TrieNode, letter_list: list[str], min_length: int
) -> None:
    """
    Handle command line letters: Find matches for each set of letters,
    print them, and return.

    :param trie: the loaded dictionary trie
    :param letter_list: the list of strings to process
    """
    header_and_sep = len(letter_list) > 1
    for letters in letter_list:
        try:
            check_string(letters, min_length)
        except ValueError as e:
            print(f'Skipping "{letters}": {e}')
            continue

        if header_and_sep:
            print(multiple_matches_header(letters))
            print(letters)
            print()

        show_matches(trie.find_matches(letters, min_length))
        if header_and_sep:
            print("-" * 50)


def load_config_file(config_path: Path, must_exist: bool) -> Params:
    """
    Load the configuration file, if it exists. If it doesn't exist and
    must_exist is True, raise an exception. (This is the case when the
    configuration file is specified on the command line.) Otherwise, if it
    doesn't exist, return a default Params object. If it does exist, load
    and validate the values, and return a Params object.

    :param config_path: the path to the configuration file
    :param must_exist: whether the configuration file must exist

    :return: a Params object
    """
    if not config_path.exists():
        if must_exist:
            raise click.ClickException(
                f"Configuration file {config_path} not found."
            )
        config_dict = {"findwords": {}}
    else:
        with open(config_path, mode="rb") as f:
            config_dict = tomllib.load(f)

    if (findwords := config_dict.get("findwords")) is None:
        raise click.ClickException(
            f'Configuration file "{config_path}" is missing the "findwords" '
            "section."
        )

    def get_path(d: dict[str, Any], key: str, default: Path) -> Path:
        """
        Get a path from a dictionary, expanding it as necessary. (e.g.,
        "~" is expanded to the current user's home directory)
        """
        path = d.get(key)
        if path is not None:
            path = Path(path).expanduser()
        else:
            path = default

        return path

    return Params(
        dictionary=get_path(findwords, "dictionary", DEFAULT_DICTIONARY),
        history_path=get_path(findwords, "history", DEFAULT_HISTORY_FILE),
        min_length=findwords.get("min_length", DEFAULT_MIN_LENGTH),
        verbose=findwords.get("verbose", False),
    )


# pylint: disable=unused-argument
def validate_min_length(ctx: click.Context, param: str, value: int) -> int:
    """
    Click callback to ensure the minimum length is a positive integer.
    """
    if value is not None:
        if value <= 0:
            raise click.BadParameter(f"{param} must be a positive integer.")

    return value


@click.command(
    name=NAME,
    context_settings=CLICK_CONTEXT_SETTINGS,
    epilog=f"Default configuration file: {DEFAULT_CONFIG_FILE}",
)
@click.option(
    "-c",
    "--config",
    type=click.Path(exists=True, dir_okay=False),
    help="Path to optional configuration file."
)
@click.option(
    "-d",
    "--dictionary",
    type=click.Path(exists=True, dir_okay=False),
    envvar="FINDWORDS_DICTIONARY",
    help="Path to dictionary to load and use. If not specified, the "
    "FINDWORDS_DICTIONARY environment variable is consulted. If that's "
    f'empty, "{DEFAULT_DICTIONARY}" is used.',
)
@click.option(
    "-H",
    "--history",
    type=click.Path(dir_okay=False),
    envvar="FINDWORDS_HISTORY",
    help="Path to readline history file. Ignored unless no words are specified "
    "on the command line (i.e., ignored in non-interactive mode). If not "
    "specified on the command line or via the FINDWORDS_HISTORY environment "
    "variable, the optional configuration file is used. If not specified, "
    f'"{DEFAULT_HISTORY_FILE}" is used.'
)
@click.option(
    "-m",
    "--min-length",
    type=int,
    callback=validate_min_length,
    help="Minimum length of words to find. Must be a positive number. If not "
    "specified and not in the configuration file, the default is "
    f"{DEFAULT_MIN_LENGTH}"
)
@click.version_option(version=VERSION)
@click.option(
    "-v",
    "--verbose",
    is_flag=True,
    help='Emit initialization messages (like "Loading dictionary") on startup.',
)
@click.argument("letter_list", nargs=-1, metavar="letters")
# pylint: disable=too-many-arguments,too-many-positional-arguments
def main(
    config: str | None,
    dictionary: str | None,
    letter_list: list[str],
    history: str | None,
    min_length: int | None,
    verbose: bool,
) -> None:
    """
    Given one or more strings of letters, find and display all the words that
    can be made from those letters. Note: Only ASCII characters are currently
    supported. If no words are specified on the command line, findwords prompts
    interactively for them, using readline().

    The dictionary is loaded from a file, which is assumed to be a list of
    words, one per line.

    You can specify default values for the command line options via a
    configuration file. If you specify the path to the configuration file,
    it must exist. If you don't specify a configuration file, findwords will
    look for a default configuration and load it if it exists. Environment
    variables, where applicable, override configuration file settings.
    Command line options override both environment variables and configuration
    file settings.
    """
    def apply_command_line_params(params: Params) -> Params:
        """
        Update a Params object with the command line parameters, where
        applicable, overriding the defaults.
        """
        if dictionary is not None:
            params = dataclasses.replace(
                params, dictionary=Path(dictionary).expanduser()
            )

        if history is not None:
            params = dataclasses.replace(
                params, dictionary=Path(history).expanduser()
            )

        if min_length is not None:
            params = dataclasses.replace(params, min_length=min_length)

        # With verbose, we only apply the command line setting if it's True.
        if verbose and not params.verbose:
            params = dataclasses.replace(params, verbose=True)

        return params


    # pylint: disable=global-statement
    global verbose_msg
    if not verbose:
        # pylint: disable=unnecessary-lambda-assignment
        verbose_msg = lambda _: None

    if config is not None:
        config_must_exist = True
        config_path = Path(config)
    else:
        config_must_exist = False
        config_path = DEFAULT_CONFIG_FILE

    params = load_config_file(config_path, config_must_exist)

    # Command line options override configuration file settings, if present.
    params = apply_command_line_params(
        load_config_file(config_path, config_must_exist)
    )

    trie = load_dictionary(params.dictionary, params.min_length)
    if len(letter_list) > 0:
        once_and_done(trie, letter_list, params.min_length)
    else:
        interactive_mode(trie, params.history_path, params.min_length)


if __name__ == "__main__":
    # pylint: disable=no-value-for-parameter
    main()
