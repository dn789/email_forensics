from pathlib import Path
import json
from typing import Any, Callable
import re


def json_default(obj: Any):
    if hasattr(obj, '__iter__'):
        return list(obj)
    else:
        return str(obj)


def dump_json(obj: Any, path: Path | str, default: Callable = json_default) -> None:
    if not isinstance(path, Path):
        path = Path(path)
    try:
        with path.open('w', encoding='utf-8') as f:
            json.dump(obj, f, default=default)
    except UnicodeDecodeError:
        return


def load_json(path: Path | str) -> dict[str, Any]:
    if not isinstance(path, Path):
        path = Path(path)
    try:
        with path.open(encoding='utf-8') as f:
            return json.load(f)
    except (UnicodeDecodeError, FileNotFoundError):
        return {}


def find_email_addresses(text: str) -> set[str]:
    addresses = re.findall(
        r'[a-zA-z0-9\.\_\-]+@[a-zA-z0-9\.\_\-]+\.[a-zA-z0-9]+', text)
    return set([a.lower() for a in addresses if a])


def find_urls(text: str) -> set[str]:
    """Returns dict of domain, list of urls of domain. """
    url_pattern = r"(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:'\".,<>?«»“”‘’]))"
    urls = re.findall(url_pattern, text)
    return set([''.join(url).lower() for url in urls])


def find_phone_nums(text: str) -> list[str]:
    phone_num_pattern = r'[\+]?[(]?[0-9]{3}[)]?[-\s\.]?[0-9]{3}[-\s\.]?[0-9]{4,6}'
    return re.findall(phone_num_pattern, text)


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


def get_sender(email: dict) -> tuple[str, str]:
    """Returns (name, address)"""
    name = email.get('senderName', '')
    address = email.get('senderEmailAddress', '').lower()
    if not address:
        if match := find_email_addresses(name):
            match, = match
            address = match[0].lower()
    return name, address


def get_recipients(email: dict) -> list[tuple[str | None, str]]:
    "Returns (name, address)"
    recipients = []
    for r in email.get('recipients', []):
        recipients.append([None, r['smtpAddress'].lower()])
    if len(recipients) == 1:
        recipients[0][0] = email.get('displayTo')
    return recipients


def get_body_text(doc_path: str) -> str | None:
    return load_json(doc_path).get('bodyText')
