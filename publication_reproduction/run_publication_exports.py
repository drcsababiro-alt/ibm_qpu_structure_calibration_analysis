#!/usr/bin/env python3
"""Build figure data, display-table summaries, figures and checks from archived results."""
import subprocess,sys
from pathlib import Path
folder=Path(__file__).resolve().parent
for script in ['prepare_figure_data.py','prepare_tables.py','generate_figures.py','verify_outputs.py']:
 print('Running',script,flush=True)
 subprocess.run([sys.executable,str(folder/script)],check=True)
