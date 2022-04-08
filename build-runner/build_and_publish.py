#!/usr/bin/python3

import os
import tempfile
import sys
import shutil
import subprocess
from argparse import ArgumentParser
import yaml
try:
    from yaml import CLoader as Loader
except ImportError:
    from yaml import Loader
from jupyter_book.cli.main import build
import jupytext

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
del sys.argv[1:]

remote_repo = f'https://github.com/{options.account}/qc-workbook'
build_path = os.path.join(options.target, '_build')

# Clone the repository first, if required (WARNING: This wipes out everything in the source directory!)

if options.checkout:
    workdir = os.path.dirname(options.source)
    shutil.rmtree(workdir, ignore_errors=True)
    os.chdir(os.path.dirname(workdir))
    # Can think about installing gitpython if needed
    subprocess.Popen(['git', 'clone', '-b', options.branch, remote_repo]).wait()
    
# Add the source directory to PYTHONPATH

try:
    os.environ['PYTHONPATH'] += f':{options.source}'
except KeyError:
    os.environ['PYTHONPATH'] = options.source
    
# Wipe out the build area if required

if options.clean:
    shutil.rmtree(build_path, ignore_errors=True)
    
# Build the book
    
full_source_path = os.path.join(options.source, options.lang)
        
with tempfile.TemporaryDirectory() as temp_home:
    # Move HOME so qiskit won't load the IBMQ credentials
    current_home = os.environ['HOME']
    os.environ['HOME'] = temp_home
    
    # Also write the build configuration file here (update the repository parameter to remote_repo)
    with open(os.path.join(full_source_path, '_config.yml')) as src:
        config = yaml.load(src, Loader)
        try:
            config['repository']['url'] = remote_repo
            config['repository']['branch'] = options.branch
        except KeyError:
            pass

    config_path = os.path.join(current_home, '_config.yml')
    
    with open(config_path, 'w') as out:
        out.write(yaml.dump(config))
    
    try:
        build.callback(path_source=full_source_path,
                       path_output=options.target,
                       config=config_path,
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

    finally:
        os.environ['HOME'] = current_home
        
# Cleanup

if not options.keep_reports:
    try:
        shutil.rmtree(os.path.join(build_path, 'html', 'reports'))
    except:
        pass

    try:
        os.remove(os.path.join(build_path, 'html', '.buildinfo'))
    except:
        pass
