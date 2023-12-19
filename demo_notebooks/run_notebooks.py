import papermill as pm
from glob import glob

# use argparse to get the notebook name pattern
import argparse
parser = argparse.ArgumentParser()

# add help text
parser.add_argument("-p", "--pattern", help="pattern for notebook name")
args = parser.parse_args()
print(f'Notebook pattern: {args.pattern}')

# if no pattern is given, run all notebooks
if args.pattern is None:
    args.pattern = '*.ipynb'


# run all notebooks matching the pattern
for nb in glob(args.pattern):
    print(f'Running notebook: {nb}')
    pm.execute_notebook(
        input_path=nb,
        output_path=nb,
    )


