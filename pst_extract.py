"""
Run with input (pst file path or folder containing them) and output folder as 
command-line arguments. Will create  output folder if it doesn't exist.
"""

import argparse
import json
import os

from bs4 import BeautifulSoup
import pypff


def get_messages_from_pst(pst_input, output_folder):
    """
    Makes JSON files of emails in output_folder.  
    ----------
    pst_path : str
    output_folder : str
    """
    errors = []
    if not os.path.isdir(output_folder):
        os.makedirs(output_folder, exist_ok=True)
    to_process = []
    if os.path.isfile(pst_input):
        to_process.append(pst_input)
    else:
        for f in os.listdir(pst_input):
            to_process.append(os.path.join(pst_input, f))
    for i, f in enumerate(to_process):
        try:
            file_ = pypff.open(f)
            root = file_.get_root_folder()
            for x in root.sub_items:
                walk_folder_for_messages(
                    x, output_folder=output_folder, pst_id=i)
        except OSError:
            errors.append(f)
    if errors:
        print(f'List of unopened files saved to: errors_for_{output_folder}')
        with open(f'errors_for_{output_folder}', 'w') as f_:
            f_.write('\n'.join(errors))


def walk_folder_for_messages(folder, output_folder, pst_id):
    for i in folder.sub_items:
        if type(i) == pypff.message:
            message = {}
            if i.transport_headers:
                headers = i.transport_headers.split('\r\n')
                for header in headers:
                    if header:
                        try:
                            k, v = header.split(':', 1)
                            message[k] = v.strip()
                        except ValueError:
                            pass

            if html := i.html_body:
                message['body'] = BeautifulSoup(html, 'lxml').text
            elif plain_text := i.plain_text_body:
                message['body'] = plain_text.decode(errors='ignore')
            elif rtf := i.plain_text_body:
                message['body'] = rtf.decode(errors='ignore')
            else:
                message['body'] = ''

            with open(os.path.join(output_folder, f'{pst_id}-{i.identifier}.json'), 'w', encoding='utf-8') as f:
                json.dump(message, f)
        elif type(i) == pypff.folder:
            walk_folder_for_messages(
                i, output_folder=output_folder, pst_id=pst_id)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="""Get emails from pst files: 
        pst_extract.py <input> <output_folder>.
        Input can be pst file path or folder containg psts.   
        Will create output folder if it doesn't exist""")

    parser.add_argument('input', help='pst file or folder containing them')
    parser.add_argument('output_folder', help='output folder')
    args = parser.parse_args()

    get_messages_from_pst(args.input, args.output_folder)
