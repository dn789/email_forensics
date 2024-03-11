from pathlib import Path
import subprocess

from .process_eml_files import process_folder


def preprocess(pst_input: Path, output: Path):
    folders_to_process = []
    psts_to_process = []

    if pst_input.is_file():
        psts_to_process.append(pst_input)

    else:
        for path in pst_input.iterdir():
            if path.is_file() and path.suffix == '.pst':
                psts_to_process.append(path)
            elif path.is_dir():
                folders_to_process.append(path)

    for pst in psts_to_process:
        subprocess.run(
            ['node', 'preprocess/process_pst_js/process_pst.js', pst, output])

    for folder in folders_to_process:
        process_folder(folder, output)
