__author__ = 'Ryan J McLaughlin'

"""
SABer command line.
"""
import argparse
import logging
import sys

from commands import (info, recruit)

usage = """
saber <command> [<args>]
** Commands include:
recruit        Recruit environmental reads to reference SAG(s).
** Other commands:
info           Display SABer version and other information.
help           Return this message.
Use '-h' to get subcommand-specific help, e.g.
"""


def main():
    print(1)
    commands = {"recruit": recruit,
                "info": info}
    print(2)
    parser = argparse.ArgumentParser(description='Recruit environmental reads to reference SAG(s).',
                                     add_help=False
                                     )
    print(3)
    parser.add_argument('command', nargs='?')
    print(4)
    input_cmd = sys.argv[1:2]
    if input_cmd == ['-h']:
        input_cmd = ['help']
    print(5)
    args = parser.parse_args(input_cmd)
    print(6)
    if (not args.command) | (args.command == 'help'):
        sys.stderr.write(usage)
        sys.exit(1)
    elif args.command not in commands:
        logging.error('Unrecognized command')
        sys.stderr.write(usage)
        sys.exit(1)
    print(7)

    cmd = commands.get(args.command)
    print(8)
    cmd(sys.argv[2:])
    logging.info("SABer has finished successfully.\n")


if __name__ == '__main__':
    main()
