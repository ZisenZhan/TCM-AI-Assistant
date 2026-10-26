"""Command entry point shared by Python and run.sh."""
from pathlib import Path
import runpy
import sys


COMMANDS = {
    'prepare': 'data.prepare',
    'train': 'modeling.train',
    'predict': 'modeling.predict',
    'evaluate': 'analysis.evaluate',
    'rank': 'analysis.rank',
    'reproduce': 'analysis.reproduce',
}


def main():
    if len(sys.argv) < 2 or sys.argv[1] not in COMMANDS:
        raise SystemExit('Usage: python3 -m cap_mpnn {prepare|train|predict|evaluate|rank|reproduce} ...')
    if not __package__:
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    command = sys.argv.pop(1)
    runpy.run_module(f'cap_mpnn.{COMMANDS[command]}', run_name='__main__', alter_sys=True)


if __name__ == '__main__':
    main()
