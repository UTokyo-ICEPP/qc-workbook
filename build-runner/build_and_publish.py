#!/usr/bin/python3

import os
import sys
import shutil
import subprocess
from argparse import ArgumentParser
from jupyter_book.cli.main import build

parser = ArgumentParser(description='Build and publish qc-workbook.')
parser.add_argument('--checkout', '-k', action='store_true', dest='checkout', help='Checkout the source files from github.')
parser.add_argument('--account', '-a', metavar='ACCOUNT', dest='account', default='UTokyo-ICEPP', help='Github account of the source repository.')
parser.add_argument('--branch', '-b', metavar='BRANCH', dest='branch', default='master', help='Branch from which to build the website.')
parser.add_argument('--source', '-i', metavar='PATH', dest='source', default='/tmp/qc-workbook/source', help='Source directory.')
parser.add_argument('--lang', '-l', metavar='LANG', dest='lang', default='ja', help='Workbook language.')
parser.add_argument('--target', '-o', metavar='PATH', dest='target', default='/tmp/qc-workbook/build', help='Build directory.')
parser.add_argument('--clean', '-c', action='store_true', dest='clean', help='Clean the target directory before build.')
parser.add_argument('--keep-reports', '-r', action='store_true', dest='keep_reports', help='Keep the reports directory and .buildinfo file.')
options = parser.parse_args()
sys.argv = []

if options.checkout:
    workdir = os.path.dirname(options.source)
    shutil.rmtree(workdir, ignore_errors=True)
    os.chdir(os.path.dirname(workdir))
    # Can think about installing gitpython if needed
    subprocess.Popen(['git', 'clone', '-b', options.branch, f'https://github.com/{options.account}/qc-workbook']).wait()

try:
    os.environ['PYTHONPATH'] += f':{options.source}'
except KeyError:
    os.environ['PYTHONPATH'] = options.source

if options.clean:
    shutil.rmtree(os.path.join(options.target, '_build'))

build.callback(path_source=os.path.join(options.source, options.lang),
               path_output=options.target,
               config=None,
               toc=None,
               warningiserror=False,
               nitpick=False,
               keep_going=False,
               freshenv=False,
               builder='html',
               custom_builder=None,
               verbose=0,
               quiet=0,
               individualpages=False)

if not options.keep_reports:
    try:
        shutil.rmtree(os.path.join(options.target, '_build', 'html', 'reports'))
    except:
        pass

    try:
        os.remove(os.path.join(options.target, '_build', 'html', '.buildinfo'))
    except:
        pass
