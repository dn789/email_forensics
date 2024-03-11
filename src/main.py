from collections import namedtuple
from pathlib import Path

from lang_models.semantic import SemanticModel
from preprocess import preprocess
from project import Project
from utils.io import dump_json, load_json


def main(config_path: str | Path):

    config = load_json(config_path)

    input_folder_or_file = Path(config['input'])
    project_folder = Path(config['project_folder'])
    checklist_path = project_folder / 'checklist.json'
    org_docs_folder = project_folder / 'org_docs'
    output_folder = project_folder / 'output'

    checklist = load_json(checklist_path) if checklist_path.is_file() else {}

    # Extracts PST contents / parses .eml files
    if not checklist.get('preprocess'):
        preprocess.preprocess(input_folder_or_file, project_folder)
        checklist['preprocess'] = True
        dump_json(checklist, checklist_path)

    semantic_model = SemanticModel()

    pst_analyzer = Project(
        org_docs_folder, output_folder, **config['pst_analyzer_args'])


if __name__ == '__main__':

    config_path = 'config.json'
    main(config_path)
