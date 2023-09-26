"""
Run with PST folder and output folder as command-line arguments. Will create
output folder if it doesn't exist.
"""

import argparse
import json
import os

import pypff


def get_messages_from_pst(pst_path, output_folder):
    """
    Makes JSON files of emails in output_folder.  
    ----------
    pst_path : str
    output_folder : str
    """
    if not os.path.isdir(output_folder):
        os.makedirs(output_folder, exist_ok=True)
    file_ = pypff.open(pst_path)
    root = file_.get_root_folder()
    for x in root.sub_items:
        walk_folder_for_messages(
            x, output_folder=output_folder)


def walk_folder_for_messages(folder, output_folder):
    for i in folder.sub_items:
        if type(i) == pypff.message:
            message = {}
            if i.transport_headers:
                headers = i.transport_headers.split('\r\n')
                for header in headers:
                    if header:
                        k, v = header.split(':', 1)
                        message[k] = v.strip()
            message['body'] = i.plain_text_body.decode()
            with open(os.path.join(output_folder, f'{i.identifier}.json'), 'w', encoding='utf-8') as f:
                json.dump(message, f)
        elif type(i) == pypff.folder:
            walk_folder_for_messages(
                i, output_folder=output_folder)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="""Get emails from PST files: 
        pst_extract.py <pst_folder> <output_folder>. 
        Will create output folder if it doesn't exist""")
    parser.add_argument('pst_folder', help='folder with PST files')
    parser.add_argument('output_folder', help='folder with PST files')
    args = parser.parse_args()
    get_messages_from_pst(args.pst_folder, args.output_folder)
