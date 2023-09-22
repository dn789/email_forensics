import json
import os

from bs4 import BeautifulSoup
from nltk.corpus import wordnet
import pypff


def get_account_password_emails(email_dir):
    account_pw_emails = []
    for root, dirs, files in os.walk(email_dir):
        for file in files:
            try:
                path = os.path.join(root, file)
                email = open(path, encoding='utf-8').read()
                text = get_email_text_and_subject(email)['main_text']
                words = [x.strip('.') for x in text.split()]
                if len(words) <= 500:
                    if ('password' in words or 'account' in words) and ('recovery' in words or 'reset' in words):
                        account_pw_emails.append(path)
            except UnicodeDecodeError:
                pass
    open('emails_with_keywords.txt', 'w').write('\n'.join(account_pw_emails))


def get_email_text_and_subject(path, combine=True, stripHTML=True):
    lines = open(path, encoding='utf-8').read().split('\n')
    start_index, end_index = None, None
    subject = None
    for index, line in enumerate(lines):
        if line.startswith('Subject:') and subject is None:
            subject = line.split(':', 1)[1].strip()
        if not line and start_index is None:
            start_index = index
        elif '-----Original Message-----' in line:
            end_index = index
            break

    text = '\n'.join(lines[start_index:end_index]).strip()
    if stripHTML:
        text = BeautifulSoup(text, 'html.parser').text
    if combine:
        return f'{subject}\n{text}'
    return {'main_text': text, 'subject': subject}


# pst files

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


def make_email_dict_from_string(email):
    """Makes dict of headers, body from text string (eg. from Enron corpus)."""
    email_dict = {}
    header_lines, body = email.split('\n\n', 1)
    header_lines = header_lines.split('\n')
    headers = []
    for line in header_lines:
        if line.startswith('\t') or line.startswith(' '):
            headers[-1] = headers[-1] + f' {line.strip()}'
        else:
            headers.append(line)
    for header in headers:
        k, v = header.split(':', 1)
        email_dict[k] = v.strip()
    email_dict['body'] = body
    return email_dict


# email = open('test_emails/10041').read()
# print(email)
# print(make_email_dict_from_string(email))
