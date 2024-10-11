import atexit
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
from typing import Self, Callable, Sequence, Tuple

import art
import click
from termcolor import colored


NAME = "findwords"
VERSION = "1.0.1"
CLICK_CONTEXT_SETTINGS = dict(help_option_names=["-h", "--help"])
HISTORY_LENGTH = 10_000
# Note that Python's readline library can be based on GNU Readline
# or the BSD Editline library, and it's not selectable. It's whatever
# has been compiled in. They use different initialization files, so we'll
# load whichever one is appropriate.
EDITLINE_BINDINGS_FILE = Path("~/.editrc").expanduser()
READLINE_BINDINGS_FILE = Path("~/.inputrc").expanduser()
DEFAULT_SCREEN_WIDTH = 79
DEFAULT_HISTORY_FILE = Path("~/.findwords-history").expanduser()
DEFAULT_CONFIG_FILE = Path("~/.findwords.cfg").expanduser()
DEFAULT_DICTIONARY = Path("~/etc/findwords/dict.txt").expanduser()
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
    (InternalCommand.HISTORY.value, "Show the command and word history."),
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
    case s:
        try:
            SCREEN_WIDTH = int(s)
        except ValueError:
            SCREEN_WIDTH = DEFAULT_SCREEN_WIDTH
            print(
                "The COLUMNS environment variable has an invalid value of "
                f'"{s}". Using screen width of {DEFAULT_SCREEN_WIDTH}.'
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


def load_dictionary(dict_path: Path) -> TrieNode:
    """
    Load the dictionary into a trie.

    :param dict_path: path to the dictionary file to load

    :return: the root of the dictionary trie
    """
    root = TrieNode()

    total_loaded = 0
    unique_words: set[str] = set()
    with open(dict_path) as f:
        verbose_msg(f'Loading dictionary "{dict_path}".')
        for line in f.readlines():
            word = line.strip()
            try:
                check_string(word, 1)
            except ValueError as e:
                verbose_msg(f'*** Skipping word "{word}": {e}')
                continue

            if word in unique_words:
                continue

            unique_words.add(word)
            add_word(word, root)
            total_loaded += 1

    verbose_msg(f"Loaded {total_loaded:,} words.")

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
        verbose_msg(f"Using editline (libedit).")
        completion_binding = "bind '^I' rl_complete"
    else:
        init_file = READLINE_BINDINGS_FILE
        verbose_msg(f"Using GNU readline.")
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

        if state < len(options):
            return options[state]
        else:
            return None

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
        padded_prefix = cmd_str.ljust(prefix_width)
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
        return f"{index:5d}. {line}"

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
        print(f"*** No matches.")
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


def find_matches(letters: str, root: TrieNode, min_length: int) -> list[str]:
    """
    Given a group of letters, find all words in the dictionary that can
    be made from them.

    :param letters: the letters to check
    :param root: the root of the loaded dictionary trie

    :return: a possibly empty list of words that match
    """

    def check_nodes(letters: str, nodes: list[TrieNode]) -> list[str]:
        def remove_first(letter: str, letters: str) -> str:
            if (i := letters.index(letter)) != -1:
                return letters[:i] + letters[i + 1 :]
            else:
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
            matches.extend(check_nodes(sub_letters, child_list))

        return matches

    sorted_letters = "".join(sorted(letters.lower()))
    all_matches = check_nodes(sorted_letters, list(root.children.values()))
    return [m for m in all_matches if len(m) >= min_length]


def interactive_mode(
    trie: TrieNode, history_path: Path, min_length: int
) -> None:
    """
    No letters on command line, so prompt for successive words with
    readline. Continues prompting until Ctrl-D or InternalCommand.EXIT.

    :param trie: the loaded dictionary trie
    :param history_path: path to the history file to use
    """

    def find_matches_for_inputs(line: str) -> None:
        # Break the line up into multiple words, to allow more than one
        # word per line.
        strings = line.split()
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

            show_matches(find_matches(s, trie, min_length))

    def handle_command(line: str) -> bool:
        """
        Handle command input. Return True if user wants to exit,
        False otherwise.
        """
        if len(line) == 0:
            return False

        exit = False

        match line.strip():
            case s if len(s) == 0:
                pass
            case InternalCommand.EXIT:
                return True
            case InternalCommand.HELP:
                show_help()
            case InternalCommand.HISTORY:
                show_history()
            case s if s[0] == InternalCommand.RERUN:
                # Special case: This is a prefix, followed by a history
                # item number.
                exit = handle_history_rerun(s)
            case s if s[0] == INTERNAL_COMMAND_PREFIX:
                print(f'"{s}" is an unknown command.')
            case s:
                find_matches_for_inputs(s)

        return exit

    def handle_history_rerun(line: str) -> bool:
        """
        Parse and process the "history rerun" command, returning
        True if the user wants to exit (i.e., re-runs an exit command)
        and False otherwise.
        """
        assert line[0] == InternalCommand.RERUN.value
        history = get_full_history()

        def rerun(command: str) -> bool:
            print(f"{PROMPT}{command}")
            return handle_command(command)

        exit = False

        match line[1:]:
            case InternalCommand.RERUN.value:
                # rerun last command
                if len(history) == 0:
                    print("History is empty.")
                else:
                    exit = rerun(history[-1][1])

            case s if ALL_DIGITS.search(s) is not None:
                # rerun command n
                n = int(s)
                history_dict = dict(history)
                cmd = history_dict.get(n)
                if cmd is None:
                    print(f"There's no history item #{n}.")
                else:
                    exit = rerun(cmd)

            case _:
                print(
                    f"{InternalCommand.RERUN.value} must either be followed "
                    f"by a number or by {InternalCommand.RERUN.value}"
                )

        return exit

    init_readline(history_path)

    rng = random.SystemRandom()
    # Print the banner using a random art font and a random terminal foreground
    # color.
    print(colored(
        art.text2art(NAME, rng.choice(ART_FONTS)), rng.choice(BANNER_COLORS)
    ))
    print(f"Version {VERSION}")
    print(f"Enter one or more strings, separated by white space.")
    print(f'Type Ctrl-D or "{InternalCommand.EXIT.value}" to exit.')
    print(f'Type "{InternalCommand.HELP.value}" for help on commands.')
    print()

    while True:
        try:
            # input() automatically uses the readline library, if it's
            # been loaded.
            exit = handle_command(input(PROMPT))
            if exit:
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

        show_matches(find_matches(letters, trie, min_length))
        if header_and_sep:
            print("-" * 50)


def validate_min_length(ctx: click.Context, param: str, value: int) -> int:
    if value <= 0:
        raise click.BadParameter(f"{param} must be a positive integer.")

    return value


@click.command(name=NAME, context_settings=CLICK_CONTEXT_SETTINGS)
@click.option(
    "-d",
    "--dictionary",
    type=click.Path(exists=True, dir_okay=False),
    envvar="FINDWORDS_DICTIONARY",
    default=DEFAULT_DICTIONARY,
    show_default=True,
    help="Path to dictionary to load and use. If not specified, the "
    "FINDWORDS_DICTIONARY environment variable is consulted. If that's "
    "empty, the default is used.",
)
@click.option(
    "-H",
    "--history",
    type=click.Path(dir_okay=False),
    default=DEFAULT_HISTORY_FILE,
    show_default=True,
    help="Path to readline history file. Ignored unless no words are specified "
    "on the command line (i.e., ignored in non-interactive mode).",
)
@click.option(
    "-m",
    "--min-length",
    default=2,
    callback=validate_min_length,
    show_default=True,
    help="Minimum length of words to find. Must be a positive number.",
)
@click.version_option(version=VERSION)
@click.option(
    "-v",
    "--verbose",
    is_flag=True,
    help='Emit initialization messages (like "Loading dictionary") on startup.',
)
@click.argument("letter_list", nargs=-1, metavar="letters")
def main(
    dictionary: str,
    letter_list: list[str],
    history: str,
    min_length: int,
    verbose: bool,
) -> None:
    """
    Given one or more strings of letters, find and display all the words that
    can be made from those letters. Note: Only ASCII characters are currently
    supported. If no words are specified on the command line, findwords prompts
    interactively for them, using readline().
    """
    global verbose_msg
    if not verbose:
        verbose_msg = lambda _: None

    trie = load_dictionary(Path(dictionary))
    if len(letter_list) > 0:
        once_and_done(trie, letter_list, min_length)
    else:
        interactive_mode(trie, Path(history), min_length)


if __name__ == "__main__":
    main()
