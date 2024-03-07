from email import parser
from email.policy import default
from pathlib import Path

from utils.io import dump_json


def make_message_dict(path: Path, parser: parser.Parser) -> dict[str, str] | None:
    try:
        message_obj = parser.parse(path.open())
    except UnicodeDecodeError:
        return
    message_dict = dict(message_obj.items())
    body_text = message_obj.get_body(preferencelist=('plain'))  # type: ignore
    body_html = message_obj.get_body(preferencelist=('html'))  # type: ignore
    if body_text:
        message_dict['bodyText'] = body_text.get_content()
    if body_html:
        try:
            message_dict['bodyHTML'] = body_html.get_content()
        except LookupError:
            pass
    return message_dict


def process_folder(folder: Path, output: Path) -> None:
    if output.name != folder.name:
        output = output / folder.name
    p = parser.Parser(policy=default)
    for path in folder.rglob('*'):
        # Neeed to parse other item types
        if path.is_file() and path.suffix == '.eml':
            message_dict = make_message_dict(path, p)
            if message_dict:
                message_dict['messageClass'] = 'IPM.Note'
                dump_json(message_dict, (output /
                                         path.relative_to(folder).with_suffix('.json')))
        else:
            (output / path.relative_to(folder)).mkdir(parents=True, exist_ok=True)
