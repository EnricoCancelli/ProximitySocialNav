import argparse
from os import path as osp
import os
import glob
import subprocess
import shutil
import sys

SKYNET_BASE_DIR = '/coc/testnvme/nyokoyama3/fair/icra/exp'

def scp(src, dst):
    subprocess.check_call(f'scp -r {src} skynet:{dst}'.split())

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('exp_dir')
    args = parser.parse_args()

    exp_dir = args.exp_dir

    skynet_dst_dir = osp.join(SKYNET_BASE_DIR, osp.basename(exp_dir))

    # Make temporary directories
    os.mkdir('tmp')
    os.mkdir('tmp/slurm_files')

    # Send the entire jsons and checkpoints directories
    ckpts_dir = osp.join(exp_dir, 'checkpoints')
    jsons_dir = osp.join(exp_dir, 'jsons')

    print('Sending checkpoints')
    scp(ckpts_dir, osp.join(skynet_dst_dir, 'checkpoints'))
    print('Sending jsons')
    scp(jsons_dir, osp.join(skynet_dst_dir, 'jsons'))

    # Copy over eval scripts
    eval_scripts = glob.glob(osp.join(exp_dir, 'slurm_files/*eval.sh'))
    for p in eval_scripts:
        shutil.copyfile(p, osp.join('tmp/slurm_files', osp.basename(p)))

    # Edit the eval scripts
    replace_dict = {
        '/private/home/naokiyokoyama/qq/exp/ig/': SKYNET_BASE_DIR,
        '--partition=learnlab': (
            '--partition=overcap\n'
            '#SBATCH --account=overcap\n'
            '#SBATCH --exclude=bmo,walle,alexa'
        ),
        '\n#SBATCH --time=72:00:00': '\n',
        '\n#SBATCH --time=72:00:00': '\n',
    }
    for p in glob.glob('tmp/slurm_files/*eval.sh'):
        with open(p) as f:
            data = f.read()
        for k,v in replace_dict.items():
            data = data.replace(k, v)
        with open(p, 'w') as f:
            f.write(data)

    # Send the eval scripts
    scp('tmp/slurm_files', osp.join(skynet_dst_dir, 'slurm_files'))


if __name__=='__main__':
    try:
        main()
    except:
        e = sys.exc_info()[0]
        print('Deleting ./tmp')
        shutil.rmtree('tmp')
        print(f"Error: {e}")

    shutil.rmtree('tmp')\
