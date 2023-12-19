import papermill as pm
from glob import glob

for nb in glob('*.ipynb'):
    pm.execute_notebook(
        input_path=nb,
        output_path=nb,
    )

