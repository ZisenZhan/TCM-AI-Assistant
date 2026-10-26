"""Command entry point shared by Python and run.sh."""
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
        raise SystemExit('Usage: python3 main.py {prepare|train|predict|evaluate|rank|reproduce} ...')
    command = sys.argv.pop(1)
    runpy.run_module(COMMANDS[command], run_name='__main__', alter_sys=True)


if __name__ == '__main__':
    main()
