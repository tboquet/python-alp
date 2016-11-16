import json
import os
import subprocess
from subprocess import PIPE
import click
from docker import Client
from .appcom import _alp_dir


col_ok = 'green'
col_warn = 'yellow'
col_not_ok = 'red'
col_info = 'cyan'


def a_text(text, result, size=80):
    msg = text.ljust(int(size // 2), '.')
    msg += result.rjust(int(size // 2), '.')
    return msg


def open_config(config, verbose=False):
    _config_path = config
    if config == '-':
        _config_path = os.path.expanduser(os.path.join(_alp_dir,
                                                       'containers.json'))
    if verbose:
        click.echo(click.style('Openning {}'.format(_config_path),
                               fg=col_info))
        click.echo()
    with open(config) as data_file:
        config = json.load(data_file)
    return config


def check_container(container, running_containers, dead_containers,
                    ports_in_use, verbose):
    name = container['name']
    if verbose:
        click.echo(click.style('Check {}:'.format(name).center(80, '-'),
                               fg=col_info, bold=True))
    res = True
    not_build = 'not_build' in container
    if name in running_containers and not_build:
        click.echo(click.style(
            a_text('Already running', '{}'.format(res)), fg=col_ok))
    elif name not in running_containers and not_build:
        res = False
        color = col_not_ok
        click.echo(click.style(
            a_text('Not running', ''), fg=col_not_ok))
    elif name in dead_containers:
        click.echo(click.style(
            a_text('name already taken', ''), fg='yellow'))
        res = False

    if res is True:
        color = col_ok
    else:
        color = col_not_ok
    if verbose:
        click.echo(click.style(a_text('Name OK:', '{}'.format(res)),
                               fg=color))

    port_OK = True
    for port in ports_in_use:
        msg = a_text('WARNING:[port in use]',
                     '{}: port already in use'.format(port))
        if 'ports' in container:
            for c_port in container['ports']:
                if c_port.split(':')[0] == str(port):
                    if not not_build:
                        click.echo(click.style(msg, fg='yellow'))
                        port_OK = False

    msg = a_text('Ports OK:', '{}'.format(port_OK))
    if port_OK:
        if verbose:
            click.echo(click.style(msg, fg=col_ok))
    else:
        res = False
        click.echo(click.style(msg, fg=col_not_ok))
    click.echo('\n')
    return res


def parse_cont(container, action, volumes=None, links=None):
    container_command = []
    not_build = 'not_build' in container
    if 'NV_GPU' in container:
        container_command.append('NV_GPU={}'.format(
            container['NV_GPU']))
        container_command.append('nvidia-docker')
    else:
        container_command.append('docker')
    container_command.append(action)

    if action == 'run':
        container_command.append(container['mode'])
        if not not_build:
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
            container_command += ['--name={}'.format(container['name'])]

            container_command.append(
                container['container_name'])
    else:
        container_command.append(container['name'])
    return container_command


def get_config_names(config):
    broker = config['broker']
    results_db = config['result_db']
    model_gen_db = config['model_gen_db']
    workers = config['workers']
    controlers = config['controlers']
    names = []
    names += [broker['name']]
    names += [results_db['name']]
    names += [model_gen_db['name']]
    workers_names = [cont['name'] for cont in workers]
    controlers_names = [cont['name'] for cont in controlers]
    names += workers_names + controlers_names
    return names, workers_names, controlers_names


def build_commands(config, action, verbose):
    broker = config['broker']
    results_db = config['result_db']
    model_gen_db = config['model_gen_db']
    workers = config['workers']
    controlers = config['controlers']
    all_commands = []

    docker_client = Client('unix://var/run/docker.sock')

    running_containers = []
    dead_containers = []
    ports_in_use = []
    for container in docker_client.containers(all=True):
        if container['State'] == 'running':
            running_containers.append(container['Names'][0].replace('/', ''))
        else:
            dead_containers.append(container['Names'][0].replace('/', ''))
        for port in container['Ports']:
            if 'PublicPort' in port:
                ports_in_use.append(port['PublicPort'])

    names, workers_names, controlers_names = get_config_names(config)

    if action == 'run':
        click.echo(click.style(
            'Start containers'.center(80, '='), fg=col_info, bold=True))
        click.echo()
        if verbose:
            click.echo(click.style(
                'Check config ports'.center(80, '-'), fg=col_info, bold=True))
            click.echo()
        all_ports = []
        mess = '{} in {}'
        for cont in [broker] + [results_db] + [model_gen_db] + workers + controlers:
            c_name = cont['name']
            if 'ports' in cont:
                for port in cont['ports']:
                    parsed_port = port.split(':')[0]
                    all_ports.append(parsed_port)
                    count_p = all_ports.count(parsed_port)
                    if count_p > 1:
                        click.echo(click.style(
                            a_text('WARNING:[port conflict]',
                                   mess.format(parsed_port, c_name)),
                            fg=col_warn))
                        raise Exception('Configuration of the port'
                                        ' is not correct')
        if verbose:
            click.echo(click.style(a_text('Ports config OK:', 'True'),
                                   fg=col_warn))
            click.echo('\n')
        links = []
        links.append('--link={}'.format(model_gen_db['name']))
        links.append('--link={}'.format(broker['name']))

        # broker
        if check_container(broker, running_containers,
                           dead_containers, ports_in_use, verbose):

            not_build = 'not_build' in broker
            if not not_build:
                broker_command = parse_cont(broker, 'run')
                all_commands += [broker_command]
            scheduler_ok = True
        else:
            scheduler_ok = False

        # database results
        if check_container(results_db, running_containers,
                           dead_containers, ports_in_use, verbose):
            results_db_command = parse_cont(results_db, 'run')
            all_commands += [results_db_command]
            results_db_ok = True
        else:
            results_db_ok = False

        # database models generators
        if check_container(model_gen_db, running_containers,
                           dead_containers, ports_in_use, verbose):
            model_gen_db_command = parse_cont(model_gen_db, 'run')
            all_commands += [model_gen_db_command]
            model_gen_db_ok = True
        else:
            model_gen_db_ok = False

        # workers
        workers_commands = []
        workers_ok = True
        for worker in workers:
            if check_container(worker, running_containers,
                               dead_containers, ports_in_use, verbose):
                workers_commands.append(parse_cont(worker, 'run', links=links))
            else:
                workers_ok = False

        # controlers
        controlers_commands = []
        controlers_ok = True
        for controler in controlers:
            if check_container(controler, running_containers,
                               dead_containers, ports_in_use, verbose):
                workers_commands.append(parse_cont(controler, 'run',
                                                   links=links))
            else:
                controlers_ok = False

        all_commands += workers_commands
        all_commands += controlers_commands

        if verbose:
            click.echo(click.style('Global Check'.center(80, '='),
                                   fg=col_info, bold=True))
            color_s = col_ok if scheduler_ok else col_not_ok
            color_rd = col_ok if results_db_ok else col_not_ok
            color_mgd = col_ok if model_gen_db_ok else col_not_ok
            color_wk = col_ok if workers_ok else col_not_ok
            color_ct = col_ok if controlers_ok else col_not_ok
            click.echo(click.style(a_text('Scheduler OK:', '{}'.format(
                scheduler_ok)), fg=color_s))
            click.echo(click.style(a_text('Results db OK:', '{}'.format(
                results_db_ok)), fg=color_rd))
            click.echo(click.style(a_text('Models db OK:', '{}'.format(
                model_gen_db_ok)), fg=color_mgd))
            click.echo(click.style(a_text('Workers OK:', '{}'.format(
                workers_ok)), fg=color_wk))
            click.echo(click.style(a_text('Controlers OK:', '{}'.format(
                controlers_ok)), fg=color_ct))
            click.echo('\n')
        if not all([scheduler_ok, results_db_ok, model_gen_db_ok, workers_ok,
                    controlers_ok]):
            raise Exception('Containers configuration not ok')

    if action in ['stop', 'restart', 'remove']:
        click.echo(click.style(
            'Stop running containers'.center(80, '='), fg=col_info, bold=True))
        click.echo()
        all_commands += [['docker', 'stop'] + names]

    if action == 'restart':
        click.echo(click.style(
            'Start running containers'.center(80, '='),
            fg=col_info, bold=True))
        click.echo()
        all_commands += [['docker', 'start'] + names]

    if action == 'rm':
        click.echo(click.style(
            'Remove running containers'.center(80, '='),
            fg=col_info, bold=True))
        click.echo()
        all_commands += [['docker', 'rm'] + names]

    return all_commands


def pull_config(config, verbose=False):
    res = True
    broker = config['broker']
    results_db = config['result_db']
    model_gen_db = config['model_gen_db']
    workers = config['workers']
    controlers = config['controlers']
    for container in [broker, results_db, model_gen_db] + workers + controlers:
        command = ['docker', 'pull', container['container_name']]
        if verbose:
            click.echo(click.style(
                'Running command:', fg=col_info))
            click.echo('{}\n'.format(' '.join(command)))

        p = subprocess.Popen(' '.join(command), shell=True, stdout=PIPE,
                             stderr=PIPE)
        output, err = p.communicate()
        if verbose:
            click.echo(click.style('{}\n'.format(output)))
            click.echo()
        if err is not None:
            res = False
    return res


def action_config(config, action, verbose=False, force=False):
    res = True
    commands = build_commands(config, action, verbose)
    for command in commands:
        if verbose:
            click.echo(click.style(
                'Running command:', fg=col_info))
            click.echo('{}\n'.format(' '.join(command)))

        p = subprocess.Popen(' '.join(command), shell=True, stdout=PIPE,
                             stderr=PIPE)
        output, err = p.communicate()
        if verbose:
            click.echo(click.style('{}\n'.format(output)))
        if err is not None:
            res = False
    return res


class Conf(object):

    def __init__(self):
        self.verbose = False


pass_config = click.make_pass_decorator(Conf, ensure=True)
