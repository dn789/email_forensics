import os
import json
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
    return set([a.lower() for a in addresses if a])


def find_urls(string):
    """Returns dict of domain, list of urls of domain. """
    url_pattern = r"(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:'\".,<>?«»“”‘’]))"
    urls = re.findall(url_pattern, string)
    return set([''.join(url).lower() for url in urls])


def find_phone_nums(string):
    phone_num_pattern = r'[\+]?[(]?[0-9]{3}[)]?[-\s\.]?[0-9]{3}[-\s\.]?[0-9]{4,6}'
    return re.findall(phone_num_pattern, string)


# def get_sender(email):
#     name = email.get('senderName', '')
#     address = email.get('senderEmailAddress', '').lower()
#     if not address:
#         if match := find_email_addresses(name):
#             address = match[0].lower()
#     return (name, address)


# def get_recipients(email):
#     recipients = set()
#     r_names, r_addrs = [], []
#     for k in ('displayTo', 'displayCC'):
#         if v := email.get(k):
#             r_names.extend([name.strip() for name in v.split(';')])
#     for r in email.get('recipients', []):
#         r_addrs.append(r['smtpAddress'].lower())
#     r_names.extend([None for x in range(len(r_addrs) - len(r_names))])
#     recipients.update([(name, addr)
#                        for name, addr in zip(r_names, r_addrs)])
#     return recipients

def get_sender(email):
    """Returns (name, address)"""
    name = email.get('senderName', '')
    address = email.get('senderEmailAddress', '').lower()
    if not address:
        if match := find_email_addresses(name):
            address = match[0].lower()
    return name, address


def get_recipients(email):
    "Returns (name, address)"
    recipients = []
    for r in email.get('recipients', []):
        recipients.append([None, r['smtpAddress'].lower()])
    if len(recipients) == 1:
        recipients[0][0] = email.get('displayTo')
    return recipients
