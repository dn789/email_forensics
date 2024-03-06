"""Functions for PST items"""
from pathlib import Path, PurePath
from typing import Any
import re

from utils.io import load_json, dump_json


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
    name = email.get('senderName', '')
    addr = email.get('senderEmailAddress', '').lower()
    if not addr:
        if match := find_email_addresses(name):
            match, = match
            addr = match[0].lower()
    return {'name': name, 'addr': addr}


def get_recipients(email: dict[str, Any]) -> list[dict[str, str | None]]:
    recipients = []
    for r in email.get('recipients', []):
        recipients.append({'name': None, 'addr': r['smtpAddress'].lower()})
    if len(recipients) == 1:
        recipients[0]['name'] = email.get('displayTo')
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
    email_d['bodyTextPreprocssed'] = body_text
    dump_json(email_d, doc_path)


def find_email_addresses(text: str) -> set[str]:
    addresses = re.findall(
        r'[a-zA-z0-9\.\_\-]+@[a-zA-z0-9\.\_\-]+\.[a-zA-z0-9]+', text)
    return set([a.lower() for a in addresses if a])


def find_urls(text: str) -> set[str]:
    """Returns dict of domain, list of urls of domain. """
    url_pattern = r"(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:'\".,<>?«»“”‘’]))"
    urls = re.findall(url_pattern, text)
    return set([''.join(url).lower() for url in urls])


def find_phone_nums(text: str, context: bool = False) -> list[str]:
    if context:
        phone_num_pattern = r'([\s\S]{,50})([\+]?[(]?[0-9]{3}[)]?[-\s\.]?[0-9]{3}[-\s\.]?[0-9]{4,6})([\s\S]{,50})'
        return re.findall(phone_num_pattern, text)

    else:
        phone_num_pattern = r'[\+]?[(]?[0-9]{3}[)]?[-\s\.]?[0-9]{3}[-\s\.]?[0-9]{4,6}'
        return re.findall(phone_num_pattern, text)
