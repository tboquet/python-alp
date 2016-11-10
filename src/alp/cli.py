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
import os
import json
import subprocess
from .appcom import _alp_dir


def parse_cont(container, action, volumes=None, links=None):
    container_command = []
    if 'NV_GPU' in container:
        container_command.append('NV_GPU={}'.format(
            container['NV_GPU']))
        container_command.append('nvidia-docker')
    else:
        container_command.append('docker')
    container_command.append(action)

    if action == 'run':
        container_command.append(container['mode'])
        if 'volumes' in container:
            for v in container['volumes']:
                container_command += ['-v', v]
        if volumes is not None:
            container_command.append(volumes)
        if links is not None:
            container_command += links
        if 'ports' in container:
            for p in container['ports']:
                container_command += ['-p', p]
        if 'options' in container:
            for option in container['options']:
                container_command.append(option)
        container_command += ['--name', container['name']]

        container_command.append(
            container['container_name'])
    else:
        container_command.append(container['name'])
    return container_command


def build_commands(config, action):
    broker = config['broker']
    results_db = config['result_db']
    model_gen_db = config['model_gen_db']
    workers = config['workers']
    controlers = config['controlers']
    all_commands = []

    if action == 'run':
        links = []
        links.append('--link={}'.format(model_gen_db['name']))
        links.append('--link={}'.format(broker['name']))

        # broker
        broker_command = parse_cont(broker, 'run')

        # database command
        r_db_command = parse_cont(results_db, 'run')
        m_g_db_command = parse_cont(model_gen_db, 'run')

        # workers
        workers_commands = [parse_cont(worker, action, links=links)
                            for worker in workers]
        controlers_commands = [parse_cont(controler, action, links=links)
                            for controler in controlers]
        all_commands += [broker_command] + [r_db_command] + [m_g_db_command]
        all_commands += workers_commands
        all_commands += controlers_commands
    if action in ['stop', 'restart', 'remove']:
        names = []
        names += [broker['name']]
        names += [results_db['name']]
        names += [model_gen_db['name']]
        for cont in workers + controlers:
            names += [cont['name']]

        all_commands += [['docker', 'stop'] + names]

    if action == 'restart':
        all_commands += [['docker', 'start'] + names]

    if action == 'remove':
        all_commands += [['docker', 'rm'] + names]

    return all_commands


def action_config(config, action):
    commands = build_commands(config, action)
    click.echo(commands)
    # for command in commands:
    #     p = subprocess.Popen(command,
    #                          stdout=subprocess.PIPE)
    #     output, err = p.communicate()
    #     p.kill()
    #     click.echo('Alp output: {}'.format(output))


@click.group()
def main(argv=sys.argv):
    """
    TODO: add verbose mode
    The alp command provide you with a number of options to manage alp services


    Args:
        argv (list): List of arguments

    Returns:
        int: A return code

    Does stuff.
    """

    print(argv)
    return 0


@main.command()
@click.argument('action', type=click.STRING, required=True)
@click.argument('config', type=click.Path(exists=True), required=True)
def service(action, config):
    """Subcommand to take action on services"""
    _config_path = config
    if config == '-':
        _config_path = os.path.expanduser(os.path.join(_alp_dir,
                                                       'containers.json'))
    with open(config) as data_file:
        config = json.load(data_file)
    if action == 'start':
        results = action_config(config, 'run')
    elif action == 'stop':
        results = action_config(config, 'stop')
    elif action == 'restart':
        results = action_config(config, 'restart')
    elif action == 'rm':
        results = action_config(config, 'rm')


# @cli.command()
# @click.argument('action', type=click.STRING, required=True)
# def status(action):
#     if service == 'start':
#         pass
#     elif service == 'stop':
#         pass
#     elif service == 'restart':
#         pass


# @cli.command()
# @click.argument('action', type=click.STRING, required=True)
# def print(action):
#     if service == 'start':
#         pass
#     elif service == 'stop':
#         pass
#     elif service == 'restart':
#         pass
