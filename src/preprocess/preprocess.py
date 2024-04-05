from pathlib import Path
import subprocess

from utils.io import dump_json

from .process_eml_files import process_folder


def preprocess(pst_input: Path, output: Path):
    folders_to_process = []
    psts_to_process = []

    sources_path = output / 'sources.json'
    source_d = {}

    if pst_input.is_file():
        psts_to_process.append(pst_input)

    else:
        for path in pst_input.iterdir():
            if path.is_file() and path.suffix == '.pst':
                psts_to_process.append(path)
            elif path.is_dir():
                folders_to_process.append(path)

    if not psts_to_process and not folders_to_process:
        raise ValueError(
            f'Couldn\'t find any PSTs or folders to process in {pst_input}')

    for pst in psts_to_process:
        print(f'\nExtracting contents of {pst.name}...')
        subprocess.run(
            ['node', 'preprocess/process_pst_js/process_pst.js', pst, output / pst.stem])
        source_d[pst.stem] = 'pst'

    for folder in folders_to_process:
        print(f'\nPreprocessing {folder.name}...')
        process_folder(folder, output)
        source_d[folder.__str__()] = 'folder'

    dump_json(source_d, sources_path)
