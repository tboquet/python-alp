import json
import os
from click.testing import CliRunner

from alp.cli import main


def init_test_config():
    config_path = '/root/.alp/containers_test.json'
    if os.getenv('TEST_MODE') == 'ON':  # pragma: no cover
        config_path = 'containers.json'
    if not os.path.exists(config_path):  # pragma: no cover
        runner = CliRunner()
        result = runner.invoke(main, ['genconfig',
                                      '--outdir={}'.format('.')])
    return config_path


config_path = init_test_config()


def test_status():
    runner = CliRunner()
    result = runner.invoke(main, ['status', config_path])
    assert result.exit_code == 0
    result = runner.invoke(main, ['--verbose', 'status', config_path])
    assert result.exit_code == 0


def test_service():
    runner = CliRunner()
    for cm in ['stop', 'rm', 'start', 'restart']:
        result = runner.invoke(main, ['service', cm, config_path])
        assert result.exit_code == 0
    for cm in ['stop', 'rm', 'start', 'restart']:
        result = runner.invoke(main, ['--verbose', 'service', cm, config_path])
        assert result.exit_code == 0
    for cm in ['stop', 'rm', 'start', 'restart']:
        result = runner.invoke(main, ['--verbose', 'service', '--dry_run',
                                      cm, config_path])
        assert result.exit_code == 0

def test_update():
    runner = CliRunner()
    result = runner.invoke(main, ['--verbose', 'update', config_path])
    assert result.exit_code == 0


def test_pull():
    runner = CliRunner()
    result = runner.invoke(main, ['pull', config_path])
    assert result.exit_code == 0
    result = runner.invoke(main, ['--verbose', 'pull', config_path])
    assert result.exit_code == 0


def test_gen():
    user_path = os.path.expanduser('~')
    gen_dir = os.path.join(user_path, '.alp')
    if not os.path.exists(gen_dir):
        os.makedirs(gen_dir)
    runner = CliRunner()
    result = runner.invoke(main, ['genconfig'])
    result = runner.invoke(main, ['genconfig', '--namesuf=test'])
    assert result.exit_code == 0
    result = runner.invoke(main, ['--verbose', 'genconfig',
                                  '--outdir={}'.format(gen_dir)])
    assert result.exit_code == 0
    result = runner.invoke(main, ['--verbose', 'genconfig',
                                  '--rootfolder={}'.format(gen_dir)])
    assert result.exit_code == 0
