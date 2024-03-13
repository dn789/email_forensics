"""Functions for PST items"""
from email.utils import parseaddr
from pathlib import Path, PurePath
from typing import Any
import re

from utils.io import load_json, dump_json


SPACE_CHARS = '\t\x0b\x0c\r\x1c\x1d\x1e\x1f \x85\xa0\u1680\u2000\u2001\u2002\u2003\u2004\u2005\u2006\u2007\u2008\u2009\u200a\u2028\u2029\u202f\u205f\u3000'


def get_contact(contactItem: dict) -> dict | None:
    """Gets 1st email and display name."""
    name = contactItem.get('email1DisplayName')
    if not name:
        name = contactItem.get('givenName', '') + \
            contactItem.get('surname', '')

    if email_addr := contactItem.get('email1EmailAddress'):
        contact = {'email_addr': email_addr}
        if name:
            contact['name'] = name
        return contact


def get_sender(email: dict[str, Any]) -> dict[str, str | None]:
    if from_str := email.get('From'):
        name, addr = parseaddr(from_str)
    else:
        name = email.get('senderName', '')
        addr = email.get('senderEmailAddress', '').lower()
        if not addr:
            if match := find_email_addresses(name):
                match, = match
                addr = match[0].lower()
    return {'name': name, 'addr': addr}


def get_recipients(email: dict[str, Any]) -> list[dict[str, str | None]]:
    recipients = []
    if r_list := email.get('recipients'):
        for r in r_list:
            recipients.append({'name': None, 'addr': r['smtpAddress'].lower()})
        if len(recipients) == 1:
            recipients[0]['name'] = email.get('displayTo')
    elif r_str := email.get('To'):
        for raw_addr in r_str.split(','):
            name, addr = parseaddr(raw_addr)
            recipients.append({'name': name, 'addr': addr.lower()})
    return recipients


def get_body_text(doc: Path | dict, incl_subj: bool = False, preprocessed: bool = True) -> str:
    if isinstance(doc, PurePath):
        doc = load_json(doc)
    body_key = 'bodyTextPreprocessed' if preprocessed else 'bodyText'
    body_text = doc.get(body_key, '').strip()
    if incl_subj:
        subject = doc.get('subject', '').strip()
        body_text = '\n\n'.join((subject, body_text))
    return body_text


def add_preprocessed_body_text(doc_path: Path, body_text: str) -> None:
    email_d = load_json(doc_path)
    email_d['bodyTextPreprocessed'] = body_text
    dump_json(email_d, doc_path)


def find_email_addresses(text: str) -> set[str]:
    addresses = re.findall(
        r'[a-zA-z0-9\.\_\-]+@[a-zA-z0-9\.\_\-]+\.[a-zA-z0-9]+', text)
    return set([a.lower() for a in addresses if a])


def find_urls(text: str) -> set[str]:
    url_pattern = r"(?i)\b(?:https?://www\d{0,3}[.]|www\d{0,3}[.])\S+"
    urls = re.findall(url_pattern, text)
    return set([''.join(url).lower() for url in urls])


def find_phone_nums(text: str, context: bool = False) -> list[str]:
    if context:
        phone_num_pattern = r'([\s\S]{,50})([\+]?[(]?[0-9]{3}[)]?[-\s\.]?[0-9]{3}[-\s\.]?[0-9]{4,6})([\s\S]{,50})'
        return re.findall(phone_num_pattern, text)

    else:
        phone_num_pattern = r'[\+]?[(]?[0-9]{3}[)]?[-\s\.]?[0-9]{3}[-\s\.]?[0-9]{4,6}'
        return re.findall(phone_num_pattern, text)


def check_if_folder_is_sent(path: Path) -> list[str]:
    return re.findall(r'(?:^|[^a-z])sent(?:$|[^a-z])', path.__str__(), flags=re.IGNORECASE)
