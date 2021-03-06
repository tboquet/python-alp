"""
CLI to launch ALP services
==========================
"""

import os
import click
import pandas as pd
from docker import Client
from . import __version__
from .cli_utils import a_text
from .cli_utils import action_config
from .cli_utils import banner
from .cli_utils import col_info
from .cli_utils import col_warn
from .cli_utils import gen_all_configs
from .cli_utils import get_config_names
from .cli_utils import open_config
from .cli_utils import pass_config
from .cli_utils import pull_config


@click.group()
@click.option('--verbose', is_flag=True)
@pass_config
def main(conf, verbose):
    """
    The alp command provide you with a number of options to manage alp services
    """
    docker_client = Client('unix://var/run/docker.sock')
    kernel_version = docker_client.info()['ServerVersion']
    click.echo(click.style(banner, fg=col_info, bold=True))
    click.echo(click.style('Version: {}'.format(__version__),
                           fg=col_info, bold=True))
    click.echo(click.style('Running with Docker version: {}'.format(
        kernel_version), fg=col_info, bold=True))
    click.echo(click.style('\n'))
    conf.verbose = verbose
    return 0


@main.command()
@click.option('--force', is_flag=True)
@click.option('--dry_run', is_flag=True)
@click.argument('action', type=click.STRING, required=True)
@click.argument('config', type=click.Path(exists=True), required=True)
@pass_config
def service(conf, force, dry_run, action, config):
    """Subcommand to take action on services"""
    config = open_config(config, conf.verbose)
    if action == 'start':
        results = action_config(config, 'run', conf.verbose, force=force,
                                dry_run=dry_run)
    elif action == 'stop':
        results = action_config(config, 'stop', conf.verbose, force=force,
                                dry_run=dry_run)
    elif action == 'restart':
        results = action_config(config, 'restart', conf.verbose, force=force,
                                dry_run=dry_run)
    elif action == 'rm':
        results = action_config(config, 'rm', conf.verbose, force=force,
                                dry_run=dry_run)
    else:
        raise Exception('Action must be in start, stop, restart, rm')
    return results


@main.command()
@click.argument('config', type=click.Path(exists=True), required=True)
@pass_config
def status(conf, config):
    """Get the status of the running containers"""
    config = open_config(config)
    docker_client = Client('unix://var/run/docker.sock')
    all_containers = docker_client.containers(all=True)
    running_containers = []
    running_ids = dict()

    names = get_config_names(config)
    for container in all_containers:
        name = container['Names'][0].replace('/', '')
        if name in names:
            print_cont = dict()
            print_cont['name'] = name
            print_cont['status'] = container['Status']
            print_cont['image'] = container['Image']
            print_cont['image_id'] = container['ImageID']
            running_ids[container['ImageID']] = print_cont['image']
            print_cont['ports'] = []
            if 'Ports' in container:  # pragma: no cover
                for port in container['Ports']:
                    pub_port = None
                    priv_port = None
                    if 'PublicPort' in port:
                        pub_port = port['PublicPort']
                    if 'PrivatePort' in port:
                        priv_port = port['PrivatePort']
                    if pub_port:
                        print_cont['ports'] += ['{}:{}'.format(pub_port,
                                                               priv_port)]
            running_containers.append(print_cont)
        else:  # pragma: no cover
            click.echo(click.style(
                a_text('{}'.format(name), 'not in the config'),
                fg=col_warn))

    click.echo(click.style('Running containers'.center(80, '='),
                           fg=col_info, bold=True))
    click.echo()
    for container in running_containers:
        click.echo(click.style('{}'.format(container['name']).center(80, '-'),
                               fg=col_info, bold=True))
        for k in container:
            if isinstance(container[k], list):
                container[k] = ' '.join(container[k])
            if len(container[k]) > 40:
                cut = len(container[k]) - 40
                container[k] = container[k][:cut - 3] + '...'
            click.echo(click.style(a_text(k, container[k]),
                                   fg=col_info))
        click.echo('\n')
    images = docker_client.images()

    click.echo(click.style('Images from the config'.center(80, '='),
                           fg=col_info, bold=True))
    click.echo()
    for image in images:
        if image['Id'] in running_ids:
            print_im = dict()
            print_im['name'] = '{}'.format(running_ids[image['Id']])
            print_im['created'] = pd.to_datetime(image['Created'] * 1e9)
            print_im['created'] = print_im['created'].strftime(
                '%Y-%m-%d %H:%M')
            print_im['size'] = '{:.2f}'.format(image['Size'] / 1000000000.)

            click.echo(click.style(
                '{}'.format(print_im['name']).center(80, '-'),
                fg=col_info, bold=True))
            for k in print_im:
                if isinstance(print_im[k], list):  # pragma: no cover
                    container[k] = ' '.join(print_im[k])
                if len(print_im[k]) > 40:  # pragma: no cover
                    cut = len(print_im[k]) - 40
                    container[k] = print_im[k][:cut - 3] + '...'
                click.echo(click.style(a_text(k, print_im[k]),
                                       fg=col_info))
            click.echo('\n')
    return 0


@main.command()
@click.option('--force', is_flag=True)
@click.argument('config', type=click.Path(exists=True), required=True)
@pass_config
def update(conf, config, force):
    """Pull, stop, remove and rerun all containers"""
    config = open_config(config)
    pull_config(config, conf.verbose)
    res_stop = action_config(config, 'stop', conf.verbose, force=force)
    res_rm = action_config(config, 'rm', conf.verbose, force=force)
    res_run = action_config(config, 'run', conf.verbose, force=force)
    succeeded = all([res_stop, res_rm, res_run])
    return succeeded


@main.command()
@click.argument('config', type=click.Path(exists=True), required=True)
@pass_config
def pull(conf, config):
    """Pull containers"""
    config = open_config(config)
    res = pull_config(config, conf.verbose)
    return res


@main.command()
@click.option('--outdir', type=click.Path(exists=True))
@click.option('--namesuf', type=click.STRING, default='')
@click.option('--portshift', type=click.INT, default=0)
@click.option('--rootfolder', type=click.Path(exists=True))
@click.option('--controlers', type=click.INT, default=1)
@click.option('--skworkers', type=click.INT, default=1)
@click.option('--kworkers', type=click.INT, default=1)
@click.option('--cpu', is_flag=True)
@pass_config
def genconfig(conf, outdir, namesuf, portshift, rootfolder, controlers,
              skworkers, kworkers, cpu):
    """Generates and writes configurations files in .alp"""

    if outdir is None:
        outdir = os.path.expanduser('~')
        if not os.access(outdir, os.W_OK):  # pragma: no cover
            outdir = '/tmp'
        outdir = os.path.join(outdir, '.alp')
    else:
        if not os.access(outdir, os.W_OK):  # pragma: no cover
            raise IOError('Cannot access directory')
        outdir = os.path.join(outdir, '.alp')
        if not os.path.exists(outdir):  # pragma: no cover
            os.makedirs(outdir)

    alpapp, alpdb, containers = gen_all_configs(outdir, namesuf, portshift,
                                                rootfolder, controlers,
                                                skworkers, kworkers, cpu)

    if conf.verbose:
        click.echo(click.style('Auto generated configuration:', fg=col_info))
        click.echo(click.style(a_text('    Controlers', str(controlers)),
                               fg=col_info))
        click.echo(click.style(a_text('    Sklearn workers', str(skworkers)),
                               fg=col_info))
        click.echo(click.style(a_text('    Keras workers', str(kworkers)),
                               fg=col_info))
        click.echo()

    # dump configs in .alp

    alpapp_json = os.path.join(outdir, 'alpapp.json')
    alpdb_json = os.path.join(outdir, 'alpdb.json')
    containers_json = os.path.join(outdir, 'containers.json')

    with open(alpapp_json, 'w') as f:
        f.write(alpapp)

    with open(alpdb_json, 'w') as f:
        f.write(alpdb)

    with open(containers_json, 'w') as f:
        f.write(containers)

    return 0
