"""
Module that contains the command line app.

Why does this file exist, and why not put this in __main__?

  You might be tempted to import things from __main__ later, but that will cause
  problems: the code will get executed twice:

  - When you run `python -malp` python will execute
    ``__main__.py`` as a script. That means there won't be any
    ``alp.__main__`` in ``sys.modules``.
  - When you import __main__ it will get executed again (as a module) because
    there's no ``alp.__main__`` in ``sys.modules``.

  Also see (1) from http://click.pocoo.org/5/setuptools/#setuptools-integration
"""
import sys
import click


@cli.group()
def main(argv=sys.argv):
    """
    The alp command provide you with a number of options to manage alp services


    Args:
        argv (list): List of arguments

    Returns:
        int: A return code

    Does stuff.
    """

    print(argv)
    return 0

@cli.command()
@click.argument('action', type=click.STRING, required=True,
                help="Action to take")
def service(action):
    """Subcommand to take action on services"""
    if service == 'start':
        pass
    elif service == 'stop':
        pass
    elif service == 'restart':
        pass

@cli.command()
@click.argument('action', type=click.STRING, required=True)
def service(action):
    if service == 'start':
        pass
    elif service == 'stop':
        pass
    elif service == 'restart':
        pass
