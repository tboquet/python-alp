"""
Utilities for the CLI
=====================
"""

import json
import os
import subprocess
from subprocess import PIPE
import click
from docker import Client


col_ok = 'green'
col_warn = 'yellow'
col_not_ok = 'red'
col_info = 'cyan'


banner = """

                         _____________________
                         ___    |__  /___  __ \_
                         __  /| |_  / __  /_/ /
                         _  ___ |  /___  ____/
                         /_/  |_/_____/_/

"""


def a_text(text, result, size=80):
    msg = text.ljust(int(size // 2), '.')
    msg += result.rjust(int(size // 2), '.')
    return msg


def open_config(config, verbose=False):
    _config_path = config
    if verbose:
        click.echo(click.style('Openning {}'.format(_config_path),
                               fg=col_info))
        click.echo()
    with open(config) as data_file:
        config = json.load(data_file)
    check_config(config)
    return config


def check_config(config):  # pragma: no cover
    for k in ['broker', 'model_gen_db', 'result_db', 'workers', 'controlers']:
        if k not in config:
            raise Exception('{} is missing from the config'.format(k))


def check_container(container, running_containers, dead_containers,
                    ports_in_use, verbose):
    name = container['name']
    if verbose:
        click.echo(click.style('Check {}:'.format(name).center(80, '-'),
                               fg=col_info, bold=True))
    res = True
    not_build = 'not_build' in container
    if name in running_containers and not_build:  # pragma: no cover
        click.echo(click.style(
            a_text('Already running', '{}'.format(res)), fg=col_ok))
    elif name not in running_containers and not_build:  # pragma: no cover
        res = False
        color = col_not_ok
        click.echo(click.style(
            a_text('Not running', ''), fg=col_not_ok))
    elif name in running_containers and not not_build:  # pragma: no cover
        res = False
        color = col_not_ok
        click.echo(click.style(
            a_text('Name already taken', ''), fg=col_warn))

    elif name in dead_containers:  # pragma: no cover
        click.echo(click.style(
            a_text('Name already taken', ''), fg=col_warn))
        res = False

    if res is True:
        color = col_ok
    else:  # pragma: no cover
        color = col_not_ok
    if verbose:
        click.echo(click.style(a_text('Name OK:', '{}'.format(res)),
                               fg=color))

    port_OK = True
    for port in ports_in_use:  # pragma: no cover
        msg = a_text('WARNING:[port in use]',
                     '{}: port already in use'.format(port))
        if 'ports' in container:
            for c_port in container['ports']:
                if c_port.split(':')[0] == str(port):  # pragma: no cover
                    if not not_build:
                        click.echo(click.style(msg, fg=col_warn))
                        port_OK = False

    msg = a_text('Ports OK:', '{}'.format(port_OK))
    if port_OK:
        if verbose:
            click.echo(click.style(msg, fg=col_ok))
    else:  # pragma: no cover
        res = False
        click.echo(click.style(msg, fg=col_not_ok))
    click.echo('\n')
    return res


def parse_cont(container, action, volumes=None, links=None):
    container_command = []
    not_build = 'not_build' in container
    if 'NV_GPU' in container:  # pragma: no cover
        container_command.append('NV_GPU={}'.format(
            container['NV_GPU']))
        container_command.append('nvidia-docker')
    else:
        container_command.append('docker')
    container_command.append(action)

    if action == 'run':
        container_command.append(container['mode'])
        if not not_build:  # pragma: no cover
            if 'volumes' in container:
                for v in container['volumes']:
                    container_command += ['-v', v]
            if volumes is not None:  # pragma: no cover
                container_command.append(volumes)
            if links is not None:  # pragma: no cover
                container_command += links
            if 'ports' in container:  # pragma: no cover
                for p in container['ports']:
                    container_command += ['-p', p]
            if 'options' in container:  # pragma: no cover
                for option in container['options']:
                    container_command.append(option)
            container_command += ['--name={}'.format(container['name'])]

            container_command.append(
                container['container_name'])
    else:  # pragma: no cover
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
    return names


def build_commands(config, action, verbose, dry_run):
    broker = config['broker']
    results_db = config['result_db']
    model_gen_db = config['model_gen_db']
    workers = config['workers']
    controlers = config['controlers']
    monitors = config['monitors']
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

    names = get_config_names(config)

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
                    if count_p > 1:  # pragma: no cover
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
        links.append('--link={}'.format(results_db['name']))
        links.append('--link={}'.format(broker['name']))

        # broker
        if check_container(broker, running_containers,
                           dead_containers, ports_in_use, verbose):

            not_build = 'not_build' in broker
            if not not_build:  # pragma: no cover
                broker_command = parse_cont(broker, 'run')
                all_commands += [broker_command]
            broker_ok = True
        else:  # pragma: no cover
            broker_ok = False

        # database results
        if check_container(results_db, running_containers,
                           dead_containers, ports_in_use, verbose) or dry_run:
            results_db_command = parse_cont(results_db, 'run')
            all_commands += [results_db_command]
            results_db_ok = True
        else:  # pragma: no cover
            results_db_ok = False

        # database models generators
        if check_container(model_gen_db, running_containers,
                           dead_containers, ports_in_use, verbose) or dry_run:
            model_gen_db_command = parse_cont(model_gen_db, 'run')
            all_commands += [model_gen_db_command]
            model_gen_db_ok = True
        else:  # pragma: no cover
            model_gen_db_ok = False

        # workers
        workers_commands = []
        workers_ok = True
        for worker in workers:  # pragma: no cover
            if check_container(worker, running_containers,
                               dead_containers, ports_in_use,
                               verbose) or dry_run:
                workers_commands.append(parse_cont(worker, 'run', links=links))
            else:  # pragma: no cover
                workers_ok = False

        # controlers
        controlers_commands = []
        controlers_ok = True
        for controler in controlers:  # pragma: no cover
            if check_container(controler, running_containers,
                               dead_containers, ports_in_use,
                               verbose) or dry_run:
                workers_commands.append(parse_cont(controler, 'run',
                                                   links=links))
            else:
                controlers_ok = False

        all_commands += workers_commands
        all_commands += controlers_commands

        monitors_commands = []
        monitors_ok = False
        links_monitors = []
        for monitor in monitors:
            if 'mongo' in monitor['name']:
                links_monitors.append('--link={}:mongo')
            else:
                links_monitors.append(
                    '--link={}:rabbitmq'.format(monitor['name']))
            if check_container(monitor, running_containers,
                               dead_containers, ports_in_use,
                               verbose) or dry_run:
                monitors_commands.append(parse_cont(monitor, 'run',
                                                    links=links))
            else:
                monitors_ok = False

        if verbose:
            click.echo(click.style('Global Check'.center(80, '='),
                                   fg=col_info, bold=True))
            if dry_run is False:
                color_s = col_ok if broker_ok else col_not_ok
                color_rd = col_ok if results_db_ok else col_not_ok
                color_mgd = col_ok if model_gen_db_ok else col_not_ok
                color_wk = col_ok if workers_ok else col_not_ok
                color_ct = col_ok if controlers_ok else col_not_ok
                color_mt = col_ok if monitors_ok else col_not_ok
                click.echo(click.style(a_text('Broker OK:', '{}'.format(
                    broker_ok)), fg=color_s))
                click.echo(click.style(a_text('Results db OK:', '{}'.format(
                    results_db_ok)), fg=color_rd))
                click.echo(click.style(a_text('Models db OK:', '{}'.format(
                    model_gen_db_ok)), fg=color_mgd))
                click.echo(click.style(a_text('Workers OK:', '{}'.format(
                    workers_ok)), fg=color_wk))
                click.echo(click.style(a_text('Controlers OK:', '{}'.format(
                    controlers_ok)), fg=color_ct))
                click.echo(click.style(a_text('Monitors OK:', '{}'.format(
                    controlers_ok)), fg=color_mt))
                click.echo('\n')
            else:
                click.echo(click.style(a_text('Dry run:', '{}'.format(
                    True)), fg=col_ok))
                click.echo('\n')

        check_dict = {'broker': broker_ok,
                      'results_db': results_db_ok,
                      'model_gen_db': model_gen_db_ok,
                      'workers': workers_ok,
                      'controlers': controlers_ok,
                      'monitors': monitors_ok}

        msg = ''
        for k, v in check_dict.items():
            if v is not True:  # pragma: no cover
                msg += '{} '.format(k)
        if not all([broker_ok, results_db_ok, model_gen_db_ok,
                    workers_ok, controlers_ok, monitors_ok]):
            if dry_run is False:
                raise Exception('{} configuration not ok'.format(msg))

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


def pull_config(config, verbose=False, dry_run=False):
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
        if dry_run is False:
            p = subprocess.Popen(' '.join(command), shell=True, stdout=PIPE,
                                 stderr=PIPE)
            output, err = p.communicate()
        if verbose:
            click.echo(click.style('{}\n'.format(output)))
            click.echo(click.style('{}\n'.format(err)))
            click.echo()
        if err is not None:  # pragma: no cover
            res = False
    return res


def action_config(config, action, verbose=False, force=False, dry_run=False):
    res = True
    commands = build_commands(config, action, verbose, dry_run)
    try:
        for command in commands:
            if verbose:
                click.echo(click.style(
                    'Running command:', fg=col_info))
                click.echo('{}\n'.format(' '.join(command)))

            output = None
            err = None
            if dry_run is False:
                p = subprocess.Popen(' '.join(command), shell=True, stdout=PIPE,
                                    stderr=PIPE)
                output, err = p.communicate()
            if verbose and output is not None:
                click.echo(click.style('{}\n'.format(output)))
                click.echo(click.style('{}\n'.format(err)))
            if err is not None:  # pragma: no cover
                res = False
    except KeyboardInterrupt:
        raise KeyboardInterrupt('Please check the status of the configuration')
    return res


def make_volumes(root_folder, volumes):
    t_volumes = ['{}:{}'.format(os.path.join(root_folder, v_root), c_root)
                 for v_root, c_root in volumes]
    return t_volumes


def gen_containers_config(conf_folder, name_suffix='', port_shift=0,
                          root_folder=None, controlers=1, workers_sklearn=1,
                          workers_keras=1, cpu=False):
    """ Generates containers config parsable with ALP CLI

    This function generates a JSON configuration usable with ALP CLI.
    It is possible to customize the configuration and change the names,
    the ports, root folder where the data relative to each container will be
    mounted and the number of workers and controlers.

    Args:
        name_suffix(str): the suffix to append to the name of each controler
        port_shift(int): the shift to apply to the original ports. This shift
            will be applied to the broker and the controlers.
        root_folder(str): if None, the user root directory will be used. The
            indicated directory will be used otherwise.
        controlers(int): number of controler to be run.
        workers_sklearn(int): number of scikit learn workers to be run.
        workers keras(int): number of keras workers to be run.

    .. note::

        Choose the number of controlers and workers wisely for your system
        to work properly. For now, the configuration will use the GPU 0 of the
        system. The support for multiple GPU machine is straightforward and
        will be added soon.
    """

    if len(name_suffix) > 0:
        name_suffix = '_{}'.format(name_suffix)

    config = dict()

    if root_folder is None:
        root_folder = os.path.expanduser('~')

    # broker config
    broker = dict()
    volumes = [('alpdata/rabbitmq/dev/log', '/dev/log'),
               ('alpdata/rabbitmq', '/var/lib/rabbitmq')]

    t_volumes = make_volumes(root_folder, volumes)
    broker['volumes'] = t_volumes
    ports = [(8080 + port_shift, 15672), (5672 + port_shift, 5672)]
    broker['ports'] = ['{}:{}'.format(p1, p2) for p1, p2 in ports]
    broker['name'] = 'rabbitmq_sched{}'.format(name_suffix)
    broker['container_name'] = 'rabbitmq:management'
    broker['mode'] = '-d'

    # results db config
    result_db = dict()
    volumes = [('alpdata/mongo_data/results', '/data/db')]
    t_volumes = make_volumes(root_folder, volumes)
    result_db['volumes'] = t_volumes
    result_db['name'] = 'mongo_results{}'.format(name_suffix)
    result_db['container_name'] = 'mongo'
    result_db['mode'] = '-d'

    # models db config
    model_db = dict()
    volumes = [('alpdata/mongo_data/models', '/data/db')]
    t_volumes = make_volumes(root_folder, volumes)
    model_db['volumes'] = t_volumes
    model_db['name'] = 'mongo_models{}'.format(name_suffix)
    model_db['container_name'] = 'mongo'
    model_db['mode'] = '-d'

    volumes = [('alpdata/mongo_data/models', '/data/db'),
               ('alpdata/parameters_h5', '/parameters_h5'),
               ('alpdata/data_generator', '/data_generator')]
    conf_full_folder = '{}:{}'.format(conf_folder, '/root/.alp')

    # Build workers list
    workers = []
    # Sklearn
    for i in range(workers_sklearn):
        worker = dict()
        t_volumes = make_volumes(root_folder, volumes)
        worker['volumes'] = t_volumes + [conf_full_folder]
        worker['name'] = 'sklearn_worker{}_{}'.format(name_suffix, i)
        worker['container_name'] = 'tboquet/full8c51workeralpsk'
        worker['mode'] = '-d'
        workers.append(worker)

    # Keras
    for i in range(workers_keras):
        worker = dict()
        t_volumes = make_volumes(root_folder, volumes)
        worker['volumes'] = t_volumes + [conf_full_folder]
        worker['name'] = 'keras_worker{}_{}'.format(name_suffix, i)
        image = 'tboquet/full8c51workeralp'
        if cpu:
            image += 'cpuk'
        else:
            image += 'k'
        worker['container_name'] = image
        if not cpu:
            worker['NV_GPU'] = '0'
        worker['mode'] = '-d'
        workers.append(worker)

    # Build controlers
    volumes += [('alpdata/notebooks', '/notebooks')]
    controlers_list = []
    for i in range(controlers):
        controler = dict()
        t_volumes = make_volumes(root_folder, volumes)
        controler['volumes'] = t_volumes + [conf_full_folder]
        controler['name'] = 'controler{}_{}'.format(name_suffix, i)
        controler['container_name'] = 'tboquet/full8c51controleralp'
        if not cpu:
            controler['NV_GPU'] = '0'
        controler['mode'] = '-d'
        ports = [(440 + i + port_shift, 8888)]
        controler['ports'] = ['{}:{}'.format(p1, p2) for p1, p2 in ports]
        controlers_list.append(controler)

    monitors_list = []
    mongo_results_monitor = dict()
    mongo_results_monitor['name'] = 'mongo_results_monitor'
    mongo_results_monitor['container_name'] = 'mongo-express'
    ports = [(8081 + port_shift, 8081)]
    mongo_results_monitor['ports'] = ['{}:{}'.format(p1, p2) for p1, p2 in ports]
    monitors_list.append(mongo_results_monitor)

    mongo_models_monitor = dict()
    mongo_models_monitor['name'] = 'mongo_models_monitor'
    mongo_models_monitor['container_name'] = 'mongo-express'
    ports = [(8082 + port_shift, 8081)]
    mongo_models_monitor['ports'] = ['{}:{}'.format(p1, p2) for p1, p2 in ports]
    monitors_list.append(mongo_models_monitor)

    celery_flower = dict()
    celery_flower['name'] = 'flower_monitor'
    celery_flower['container_name'] = 'tboquet/anaceflo'
    ports = [(5555 + port_shift, 5555)]
    celery_flower['ports'] = ['{}:{}'.format(p1, p2) for p1, p2 in ports]
    host_address = 'http://guest:guest@rabbitmq:15672/api/'
    command = 'celery -A alp.appcom flower --port=5555 --broker_api={}'
    celery_flower['command'] = command.format()
    monitors_list.append(celery_flower)

    config = dict()
    config['broker'] = broker
    config['result_db'] = result_db
    config['model_gen_db'] = model_db
    config['workers'] = workers
    config['controlers'] = controlers_list
    config['monitors'] = monitors_list

    return config


def gen_alpdb_config(name_suffix=''):
    if len(name_suffix) > 0:
        name_suffix = '_{}'.format(name_suffix)

    return {'db_engine': 'mongodb',
            'host_adress': 'mongo_models{}'.format(name_suffix)}


def gen_alpapp_config(name_suffix=''):
    if len(name_suffix) > 0:
        name_suffix = '_{}'.format(name_suffix)

    port = 5672
    broker_url = 'amqp://guest:guest@rabbitmq_sched{}:{}'

    config = dict()
    config['broker'] = broker_url.format(name_suffix, port)
    cont_name = 'mongo_results{}'.format(name_suffix)
    config['backend'] = 'mongodb://{}:27017'.format(cont_name)
    config['path_h5'] = '/parameters_h5'

    return config


def gen_all_configs(conf_folder, name_suffix='', port_shift=0,
                    root_folder=None, controlers=1, workers_sklearn=1,
                    workers_keras=1, cpu=False):
    alpapp = gen_alpapp_config(name_suffix)
    alpdb = gen_alpdb_config(name_suffix)
    containers = gen_containers_config(conf_folder, name_suffix=name_suffix,
                                       port_shift=port_shift,
                                       root_folder=root_folder,
                                       controlers=controlers,
                                       workers_sklearn=workers_sklearn,
                                       workers_keras=workers_keras,
                                       cpu=cpu)
    alpapp_json = json.dumps(alpapp, indent=4)
    alpdb_json = json.dumps(alpdb, indent=4)
    containers_json = json.dumps(containers, indent=4)
    return alpapp_json, alpdb_json, containers_json


class Conf(object):

    def __init__(self):
        self.verbose = False


pass_config = click.make_pass_decorator(Conf, ensure=True)
