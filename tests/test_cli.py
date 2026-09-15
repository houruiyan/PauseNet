"""Regression tests for the packaged command-line entry point."""
import subprocess
import sys

import pytest


@pytest.mark.parametrize('command', [[], ['train'], ['evaluate'],
                                     ['prepare-bigwig']])
def test_cli_help(command):
    result = subprocess.run(
        [sys.executable, '-m', 'pausenet.cli', *command, '--help'],
        capture_output=True, text=True, timeout=60,
    )
    assert result.returncode == 0, result.stderr
    assert 'usage:' in result.stdout
    assert 'Traceback' not in result.stderr


def test_visualize_command_is_removed():
    result = subprocess.run(
        [sys.executable, '-m', 'pausenet.cli', 'visualize'],
        capture_output=True, text=True, timeout=60,
    )
    assert result.returncode == 2
    assert 'invalid choice' in result.stderr
    assert 'Traceback' not in result.stderr
