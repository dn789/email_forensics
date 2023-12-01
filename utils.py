import os
import json
import os
import re


def dump_json(obj, *path, default=None):
    path = os.path.join(*path)
    try:
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(obj, f, default=default)
    except UnicodeDecodeError:
        return


def load_json(*path):
    path = os.path.join(*path)
    try:
        with open(path, encoding='utf-8') as f:
            return json.load(f)
    except UnicodeDecodeError:
        return


def find_email_addresses(string):
    addresses = re.findall(
        r'[a-zA-z0-9\.\_\-]+@[a-zA-z0-9\.\_\-]+\.[a-zA-z0-9]+', string)
    return [a.lower() for a in addresses]


def get_sender(email):
    name = email.get('senderName', '')
    address = email.get('senderEmailAddress', '').lower()
    if not address:
        if match := find_email_addresses(name):
            address = match[0].lower()
    return (name, address)


def get_recipients(email):
    recipients = set()
    r_names, r_addrs = [], []
    for k in ('displayTo', 'displayCC'):
        if v := email.get(k):
            r_names.extend([name.strip() for name in v.split(';')])
    for r in email.get('recipients', []):
        r_addrs.append(r['smtpAddress'].lower())
    r_names.extend([None for x in range(len(r_addrs) - len(r_names))])
    recipients.update([(name, addr)
                       for name, addr in zip(r_names, r_addrs)])
    return recipients
