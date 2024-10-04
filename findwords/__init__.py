import atexit
from dataclasses import dataclass, field
from pathlib import Path
import readline
import string
from termcolor import colored
from typing import Self, Callable

import click


NAME = "findwords"
VERSION = "0.0.4"
CLICK_CONTEXT_SETTINGS = dict(help_option_names=["-h", "--help"])
HISTORY_LENGTH = 10000
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
EXIT_COMMAND = ".exit"


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
       "abehlpt"

    etc.
    """

    letter: str | None = field(default=None)
    children: dict[str, Self] = field(default_factory=dict)
    words: set[str] = field(default_factory=set)


def _emit_message(s: str) -> None:
    print(s)


# Will be changed to something else if -q is specified.
msg: Callable[[str], None] = _emit_message


def valid_string(s: str) -> bool:
    """
    Determine whether a string to match is valid. Currently, only strings
    with ASCII letters are permitted.

    :param s: the string to check

    :return: True if the string is valid, False if not
    """
    for letter in s.lower():
        if letter not in string.ascii_lowercase:
            return False

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
        msg(f'Loading dictionary "{dict_path}".')
        for line in f.readlines():
            word = line.strip()
            if not valid_string(word):
                msg(f'*** Skipping word "{word}": Invalid letter(s)')
                continue

            if word in unique_words:
                continue

            unique_words.add(word)
            add_word(word, root)
            total_loaded += 1

    msg(f"Loaded {total_loaded:,} words.")

    return root


def header(s: str) -> str:
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

    # Group them by word length.
    by_length: dict[int, list[str]] = {}
    for word in matches:
        word_len = len(word)
        len_words: list[str] = by_length.get(word_len, [])
        len_words.append(word)
        by_length[word_len] = len_words

    # Print each group, sorted within the group.
    for length in sorted(by_length.keys()):
        words = sorted(by_length[length])
        for word in words:
            print(word)

        print()


def find_matches(letters: str, root: TrieNode) -> list[str]:
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
        cur_letters = letters
        # Now, recursively check each one.
        for n in nodes:
            matches.extend(list(n.words))

            # Remove the first instance of n.letter from the letters, and
            # recursively check the child nodes of this node.
            assert n.letter is not None
            cur_letters = remove_first(n.letter, cur_letters)
            child_list = list(n.children.values())
            matches.extend(check_nodes(cur_letters, child_list))

        return matches

    sorted_letters = "".join(sorted(letters.lower()))
    return check_nodes(sorted_letters, list(root.children.values()))


def init_readline_history(history_path: Path) -> None:
    """
    Load the local readline history file.

    :param history_path: Path of the history file. It doesn't have to exist.
    """
    if history_path.exists():
        msg(f'Loading history from "{history_path}".')
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
        msg(f"Using editline (libedit).")
    else:
        msg(f"Using GNU readline.")
        init_file = READLINE_BINDINGS_FILE

    if init_file.exists():
        msg(f'Loading readline bindings from "{init_file}".')
        readline.read_init_file(init_file)


def interactive_mode(trie: TrieNode, history_path: Path) -> None:
    """
    No letters on command line, so prompt for successive words with
    readline. Continues prompting until Ctrl-D or ".exit".

    :param trie: the loaded dictionary trie
    :param history_path: path to the history file to use
    """
    init_readline_history(history_path)
    init_readline_bindings()
    print(f"{NAME}, version {VERSION}")
    print(f"Enter one or more strings, separated by white space.")
    print(f"Type Ctrl-D or {EXIT_COMMAND} to exit.\n")

    while True:
        try:
            # input() automatically uses the readline library, if it's
            # been loaded.
            line = input(PROMPT).strip()

            if len(line) == 0:
                continue

            if line == EXIT_COMMAND:
                break

            # Break the line up into multiple words, to allow more than one
            # word per line.

            strings = line.split()
            use_prefix = len(strings) > 1
            for s in line.split():
                if not valid_string(s):
                    print(f'"{s}" contains non-letter characters. Ignored.')
                    continue

                if use_prefix:
                    print()
                    print(header(s))
                    print()

                show_matches(find_matches(s, trie))

        except EOFError:
            # Ctrl-D to input()
            print()
            break


def once_and_done(trie: TrieNode, letter_list: list[str]) -> None:
    """
    Handle command line letters: Find matches for each set of letters,
    print them, and return.

    :param trie: the loaded dictionary trie
    :param letter_list: the list of strings to process
    """
    header_and_sep = len(letter_list) > 1
    for letters in letter_list:
        if not valid_string(letters):
            print(f'Skipping "{letters}", as it has non-letter characters.')
            continue

        if header_and_sep:
            print(header(letters))
            print(letters)
            print()

        show_matches(find_matches(letters, trie))
        if header_and_sep:
            print("-" * 50)


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
    "-v",
    "--verbose",
    is_flag=True,
    help='Emit initialization messages (like "Loading dictionary") on startup.'
)
@click.argument("letter_list", nargs=-1, metavar="letters")
def main(
    dictionary: str,
    letter_list: list[str],
    history: str,
    verbose: bool
) -> None:
    """
    Given one or more strings of letters, find and display all the words that
    can be made from those letters. Note: Only ASCII characters are currently
    supported. If no words are specified on the command line, findwords prompts
    interactively for them, using readline().
    """
    global msg
    if not verbose:
        msg = lambda _: None

    trie = load_dictionary(Path(dictionary))
    if len(letter_list) > 0:
        once_and_done(trie, letter_list)
    else:
        interactive_mode(trie, Path(history))


if __name__ == "__main__":
    main()
