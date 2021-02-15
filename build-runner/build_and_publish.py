#!/usr/bin/python3

import os
import sys
import shutil
import subprocess
from argparse import ArgumentParser
from jupyter_book.commands import build

parser = ArgumentParser(description='Build and publish qc-workbook.')
parser.add_argument('--account', '-a', metavar='ACCOUNT', dest='account', default='UTokyo-ICEPP', help='Github account of the source repository.')
parser.add_argument('--branch', '-b', metavar='BRANCH', dest='branch', default='master', help='Branch from which to build the website.')
parser.add_argument('--no-checkout', '-n', action='store_true', dest='no_checkout', help='Don\'t checkout from github.')
parser.add_argument('--ignore-errors', '-g', action='store_true', dest='ignore_errors', help='Run build with --keep-going flag.')
parser.add_argument('--work-dir', '-w', metavar='PATH', dest='workdir', default='/tmp', help='Working area.')
options = parser.parse_args()
sys.argv = []

if not options.no_checkout:
    shutil.rmtree('{}/qc-workbook'.format(options.workdir), ignore_errors=True)
    os.chdir(options.workdir)
    # Can think about installing gitpython if needed
    subprocess.Popen(['git', 'clone', '-b', options.branch, 'https://github.com/{}/qc-workbook'.format(options.account)]).wait()

os.chdir('{}/qc-workbook'.format(options.workdir))

build.callback(path_source='source/jp',
               path_output='/build',
               config=None,
               toc=None,
               warningiserror=False,
               nitpick=False,
               keep_going=options.ignore_errors,
               freshenv=False,
               builder='html',
               custom_builder=None,
               verbose=0,
               quiet=0,
               individualpages=False)
