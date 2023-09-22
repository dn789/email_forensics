"""
Get sender-recipient pairs for emails. 
"""
from collections import Counter
import json
import os

from utils import make_email_dict_from_string


def get_email_sender_and_recipients(email):
    """Gets sender and recpients from email dict or string."""
    if type(email) == str:
        email = make_email_dict_from_string(email)
    if not email.get('From') or not email.get('To'):
        return
    sender_string = email['From']
    recepient_string = email['To']

    sender_and_recipients = {}
    for string, label, in zip((sender_string, recepient_string), ('sender', 'recipients')):
        names_emails = []
        for name_address in string.split(','):
            if len(name_address_split := name_address.split('<')) == 2:
                name, address = name_address_split
                name = name.strip('" ')
                address = address.strip('>')
            else:
                name = None
                address = name_address_split[0]
            names_emails.append({'name': name, 'address': address})
        if label == 'sender':
            names_emails = names_emails[0]
        sender_and_recipients[label] = names_emails
    return sender_and_recipients


def get_all_sender_recipient_pairs(folder, output):
    pairs_dict = Counter()
    for root, dirs, files in os.walk(folder):
        for f in files:
            path = os.path.join(root, f)
            try:
                email = open(path, encoding='utf-8').read()
            except UnicodeDecodeError:
                continue
            sender_and_recipents = get_email_sender_and_recipients(email)
            if not sender_and_recipents:
                continue
            sender = sender_and_recipents['sender']
            for recipient in sender_and_recipents['recipients']:
                k = f'{sender["address"]}\t{recipient["address"]}'
                pairs_dict[k] += 1
    json.dump(pairs_dict, open(output, 'w', encoding='utf-8'))


# get_all_sender_recipient_pairs('test_emails', 'to_from_pairs.json')
