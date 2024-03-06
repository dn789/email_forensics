"""Gets emails, urls, phone numbers, contacts, etc. from PST docs."""

from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import tldextract
from tqdm import tqdm

from utils.io import dump_json, load_json
from utils.doc import (get_sender,
                       get_recipients,
                       get_contact,
                       find_email_addresses,
                       find_phone_nums,
                       find_urls)


@dataclass
class UserInfo:
    name: str
    contacts: list[dict[str, str]] = field(default_factory=list)
    communicators: Counter = field(default_factory=Counter)
    email_addrs: set[str] = field(default_factory=set)


@dataclass
class OrgInfo:
    users: list[UserInfo] = field(default_factory=list)
    email_addrs_w_names: dict[str, set[str]] = field(default_factory=dict)
    org_email_addrs_w_names: dict[str, set[str]] = field(default_factory=dict)
    urls: set[str] = field(default_factory=set)
    org_urls: set[str] = field(default_factory=set)
    org_domains: set[str] = field(default_factory=set)
    phone_nums_w_context: dict[str, set[str]] = field(default_factory=dict)


def process_user_folder(folder: Path, org_info: OrgInfo) -> None:
    user = UserInfo(name=folder.name)
    org_info.users.append(user)
    for path in folder.rglob('*.json'):
        if path.is_file():
            process_doc(path, user, org_info)


def process_doc(path: Path, user: UserInfo, org_info: OrgInfo) -> None:
    doc = load_json(path)
    try:
        doc_type = doc['messageClass']
    except KeyError:
        print(f'Document {path} does not have a messageClass, ignoring...')
        return

    if doc_type == 'IPM.Note':
        sender = get_sender(doc)
        recipients = get_recipients(doc)
        for addr_w_name_dict in [sender] + recipients:
            name = addr_w_name_dict['name']
            addr = addr_w_name_dict['addr']
            if addr:
                org_info.email_addrs_w_names.setdefault(addr, set())
                if name:
                    org_info.email_addrs_w_names[addr].add(name)
        # Checks if this is a sent item. Might need more robust method.
        if 'Sent Items' in path.__str__():
            user.communicators.update([r['addr'] for r in recipients])
            # Get company/organization domains from sender in sent items.
            if (sender_addr := sender['addr']) and '@' in sender_addr:
                org_info.org_domains.add(sender_addr.split('@')[1])
        else:
            user.communicators[sender['addr']] += 1

    elif doc_type == 'IPM.Contact':
        if contact := get_contact(doc):
            user.contacts.append(contact)

    if body := doc.get('bodyText'):
        email_addrs = find_email_addresses(body)
        for addr in email_addrs:
            org_info.email_addrs_w_names.setdefault(addr,  set())
        org_info.urls.update(find_urls(body))
        phone_nums_w_context = find_phone_nums(body, context=True)
        for match in phone_nums_w_context:
            context1, phone_num, context2 = match
            org_info.phone_nums_w_context.setdefault(
                phone_num, set())
            org_info.phone_nums_w_context[phone_num].add(
                f'{context1} [PHONE NUM] {context2}')


def get_contact_info(folder: Path) -> OrgInfo:
    org_info = OrgInfo()
    for user_folder in tqdm(list(folder.iterdir())):
        process_user_folder(user_folder, org_info)

    usernames = set([user.name for user in org_info.users])

    # Gets org emails
    for addr, names in org_info.email_addrs_w_names.items():
        if '@' in addr:
            domain = addr.split('@')[1]
            if domain in org_info.org_domains:
                org_info.org_email_addrs_w_names[addr] = names
            elif usernames & names:
                if domain not in ('gmail.com', 'yahoo.com', 'hotmail.com'):
                    org_info.org_email_addrs_w_names[addr] = names
                    org_info.org_domains.add(domain)

    # Gets email addrs for user. Add fuzzy matching and recursive search?
    for user in org_info.users:
        for addr, names in org_info.email_addrs_w_names.items():
            if user.name in names:
                user.email_addrs.add(addr)

    # Gets org urls
    for url in org_info.urls:
        parsed = tldextract.extract(url)
        domain = parsed.domain + '.' + parsed.suffix
        if domain in org_info.org_domains:
            org_info.org_urls.add(url)

    return org_info


def save_contact_info(org_info: OrgInfo, output: Path) -> None:

    for user in org_info.users:
        user_folder = output / 'users' / user.name
        user_folder.mkdir(parents=True, exist_ok=True)
        dump_json(user.contacts, user_folder / 'contacts.json')
        dump_json(user.communicators.most_common(20),
                  user_folder / 'communicators.json')
        dump_json(user.email_addrs, user_folder / 'email_addrs.json')

    email_addrs_folder = output / 'email_addrs'
    urls_folder = output / 'urls'
    phone_nums_folder = output / 'phone_nums'
    for folder in (email_addrs_folder, urls_folder, phone_nums_folder):
        folder.mkdir(parents=True, exist_ok=True)

    dump_json(org_info.email_addrs_w_names,
              email_addrs_folder / 'email_addrs.json')
    dump_json(org_info.org_email_addrs_w_names,
              email_addrs_folder / 'org_email_addrs.json')

    dump_json(org_info.urls,
              urls_folder / 'urls.json')
    dump_json(org_info.org_urls,
              urls_folder / 'org_urls.json')

    dump_json(org_info.phone_nums_w_context,
              phone_nums_folder / 'phone_nums_w_context.json')


def get_signatures_and_save(freq_text_blocks: list[str], output: Path) -> None:
    signatures = []
    for text_block in freq_text_blocks:
        if find_email_addresses(text_block) and find_phone_nums(text_block):
            signatures.append(text_block)
    dump_json(signatures, output)


def main(folder: Path, output: Path) -> None:
    org_info = get_contact_info(folder)
    save_contact_info(org_info, output)
