"""Find api keys, passwords etc."""
from pathlib import Path
import re
import subprocess

from utils.doc_ref import DocRef
from utils.doc import get_body_text


def make_body_text_folder(doc_ref: DocRef, project_folder: Path) -> Path:
    body_text_folder = project_folder / 'body_text'
    body_text_folder.mkdir(parents=True, exist_ok=True)
    paths = doc_ref.get_paths()
    if len([path for path in body_text_folder.rglob('*') if path.is_file()]) == len(paths):
        return body_text_folder
    for path in paths:
        if path.is_file():
            body = get_body_text(path)
            rel_path = body_text_folder / \
                path.relative_to(project_folder / 'docs')
            folder = rel_path.parent
            folder.mkdir(parents=True, exist_ok=True)
            rel_path.write_text(body)
    return body_text_folder


def find_secrets_in_docs(gitleaks_path: Path, doc_ref: DocRef, project_folder: Path, config: Path | None = None) -> None:
    """Uses gitleaks to find passwords, API keys, etc. in body text.

    Args:
        gitleaks_path (Path | None, optional): Path to gitleaks executable. 
            Defaults to None.
        doc_ref (Doc_ref): doc ref.
        project_folder (Path): project folder
        config (Path | None, optional): Path to gitleaks config toml file. 
            Defaults to None.
    """
    body_text_folder = make_body_text_folder(doc_ref, project_folder)
    args = [gitleaks_path, 'detect', '--no-git', '-s', body_text_folder,
            '-r', project_folder/'output' / 'secrets.json']
    if config:
        args.extend(['--config', config])
    subprocess.run(args)


# def old_find_secrets_in_doc(doc, secrets_ref: dict[str, dict[str, str]]):
#     secrets = []
#     for k, v in secrets_ref.items():
#         secret_type = k
#         regex = v.get('regex')
#         if not regex:
#             continue
#         try:
#             if v.get('case_sensitive'):
#                 matches = re.findall(regex, doc)
#             else:
#                 matches = re.findall(regex, doc, flags=re.IGNORECASE)

#             for match in matches:
#                 secret = {
#                     'type': secret_type,
#                     'secret': match,
#                     'description': v.get('description')
#                 }
#                 secrets.append(secret)
#         except Exception:
#             print(regex)
#             raise Exception
#     return secrets
