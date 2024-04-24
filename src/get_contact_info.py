"""Gets emails, urls, phone numbers, contacts, etc. from PST docs."""

from collections import Counter
from dataclasses import dataclass, field
from json import JSONDecodeError
from pathlib import Path
from typing import Callable
import random
import re
import warnings
from ast import literal_eval

from fuzzysearch import find_near_matches
import tldextract

from clean_body_text import get_freq_deduped_matches, get_matches_in_bodies
from utils.io import dump_json, load_json
from utils.doc import (check_if_folder_is_sent,
                       get_sender,
                       get_recipients,
                       get_contact,
                       find_email_addresses,
                       find_phone_nums,
                       find_urls)
from utils.doc_ref import DocRef
from utils.paths import JOB_TITLES
from utils.messages import MULTI_USER_PROMPT


@dataclass
class UserInfo:
    name: str
    all_names: set[str] = field(default_factory=set)
    contacts: list[dict[str, str]] = field(default_factory=list)
    communicators_counter: Counter = field(default_factory=Counter)
    communicators: dict[str, dict[str, set | Counter]
                        ] = field(default_factory=dict)
    email_addrs: set[str] = field(default_factory=set)
    job_titles: list[str] = field(default_factory=list)
    signatures: list[str] = field(default_factory=list)


@dataclass
class OrgInfo:
    users: list[UserInfo] = field(default_factory=list)
    email_addrs_w_names: dict[str, set[str]] = field(default_factory=dict)
    org_email_addrs_w_names: dict[str, set[str]] = field(default_factory=dict)
    urls: set[str] = field(default_factory=set)
    org_urls: set[str] = field(default_factory=set)
    org_domains: set[str] = field(default_factory=set)
    phone_nums_w_context: dict[str, set[str]] = field(default_factory=dict)


def get_greeting(text: str) -> str:
    greeting = text.strip().split('\n')[0].strip()
    return greeting


def check_if_signature_wrapper(user_name: str) -> Callable[[str], bool | None]:
    def check_if_signature(text: str) -> bool | None:
        email_indices = find_email_addresses(text, indices_only=True)
        phone_nums = find_phone_nums(text, ignore_whitespace=True)
        if not (email_indices or phone_nums):
            return
        last_name = user_name.split()[-1]
        name_matches = find_near_matches(last_name, text, max_l_dist=2)
        for name_match in name_matches:
            name_string = re.escape(name_match.matched)
            all_name_indices = [set(range(*match.span()))
                                for match in re.finditer(rf'\b{name_string}\b', text)]
            if any(not name_indices & email_indices for name_indices in all_name_indices):  # type: ignore
                return True

    return check_if_signature


def get_job_titles(text: str, job_titles_ref: list[str]) -> list[str]:
    return sorted([job.strip() for job in job_titles_ref if re.findall(fr'\b{job}\b', text, flags=re.IGNORECASE)], key=len, reverse=True)[:2]


def get_signatures_for_user(user: UserInfo, paths: list[Path], job_titles_ref: list[str], n_docs_to_sample: int = 100,) -> None:
    random.shuffle(paths)
    paths = paths[:n_docs_to_sample]
    matches_w_paths = get_matches_in_bodies(
        paths, min_match_size=0, match_criteria=check_if_signature_wrapper(user.name))
    doc_count_threshold = int(len(paths) * .05)
    signatures = get_freq_deduped_matches(
        matches_w_paths, doc_count_threshold=doc_count_threshold, fallback_to_most_frequent=True)

    # Prioritize signatures with email and phone number
    for i, signature in enumerate(signatures):
        if i and find_email_addresses(signature) and find_phone_nums(signature):
            signatures.pop(i)
            signatures = [signature] + signatures
            break

    if signatures:
        user.signatures = signatures
        job_titles = get_job_titles(signatures[0], job_titles_ref)
        user.job_titles = job_titles


def determine_user(folder: Path, doc_ref: DocRef, from_pst: bool = True) -> list[dict[str, set[str] | str]] | None:
    users = []
    if from_pst:
        folder_paths = doc_ref.get_paths_by_user_folder(folder)
    else:
        folder_paths = doc_ref.get_paths_by_user_folder(folder, sent_only=True)
    potential_users = {}
    for path in folder_paths:
        if path.is_file():
            try:
                doc = load_json(path)
            except JSONDecodeError:
                print(f'JSONDecodeError for {path}. Skipping...')
                continue
            if from_pst:
                addr = doc.get('receivedByAddress')
                name = doc.get('receivedByName')
            else:
                sender = get_sender(doc)
                addr = sender.get('addr')
                name = sender.get('name')

            if addr:
                potential_users.setdefault(addr, {
                    'names': Counter(),
                    'count': 0
                })
                potential_users[addr]['count'] += 1
                if name:
                    potential_users[addr]['names'][name] += 1

    if not potential_users:
        return

    if len(potential_users) == 1:
        (addr, d), = potential_users.items()
        name = d['names'].most_common()[0][0] if d['names'] else addr
        user = {'addr':  addr, 'name': name, 'addrs': set([addr]), ** d}
        users.append(user)
        return users

    potential_users_l = []
    users_str_l = []
    for i, (addr, d) in enumerate(potential_users.items()):
        users_str_l.append(
            f'({i + 1}) {addr}; names: {" ".join(d["names"])}; count: {d["count"]}')
        user_d = {'addr': addr, **d}
        potential_users_l.append(user_d)
    users_str = '\n'.join(users_str_l)
    selection = input(
        f"""\n{len(potential_users)} potential users found for {folder.name}:\n{users_str}\n\n{MULTI_USER_PROMPT}"""
    )
    if selection == 'all':
        selection = [list(range(1, len(potential_users_l) + 1))]
    else:
        selection = literal_eval(selection)
    if isinstance(selection, int):
        selection = [selection]
    if not hasattr(selection, '__iter__'):
        raise ValueError('Enter an iterable, integer, or "all"')
    for int_group in selection:
        if isinstance(int_group, int):
            int_group = [int_group]
        user_items = [potential_users_l[int_ - 1]
                      for int_ in int_group]
        user_items.sort(key=lambda x: x[
            'count'], reverse=True)
        names = Counter()
        email_addrs = set()
        for user_item in user_items:
            names.update(user_item['names'])
            email_addrs.add(user_item['addr'])
        user = {
            'addr': user_items[0]['addr'],
            'name': names.most_common()[0][0] if names else user_items[0]['addr'],
            'names': names,
            'addrs': email_addrs
        }
        users.append(user)

    return users


def process_user(user_folder: Path, org_info: OrgInfo, doc_ref: DocRef) -> None:
    from_pst = doc_ref.source_dict[user_folder.name.__str__()] == 'pst'
    users = determine_user(user_folder, doc_ref, from_pst=from_pst)
    if not users:
        warnings.warn(
            f'\nCan\'t find a user for {user_folder.name}. Processing without getting user-specific info... ')
        for path in user_folder.rglob('*'):
            if path.is_file():
                process_doc(path, None, org_info)
        return

    for user_d in users:
        user = UserInfo(name=user_d["name"])  # type: ignore
        user.all_names.update(user_d['names'])
        user.email_addrs.update(addr.lower() for addr in user_d['addrs'])
        org_info.users.append(user)

        print(
            f'\nProcessing folder: {user_folder.name} (User: {user_d["name"]})...')

        include_contacts = False if len(users) > 1 else True
        paths_to_process = doc_ref.get_paths_with_email_addrs(
            user.email_addrs, user_folder, include_contacts=include_contacts)

        for path in paths_to_process:
            process_doc(path, user, org_info)

        print(f'\nGetting signatures...')
        job_titles_ref = open(JOB_TITLES).read().split('\n')
        sent_paths_to_process = [path for path,
                                 d in paths_to_process.items() if d['sent']]
        get_signatures_for_user(
            user, sent_paths_to_process, job_titles_ref)


def process_doc(path: Path, user: UserInfo | None, org_info: OrgInfo) -> None:
    doc = load_json(path)
    try:
        doc_type = doc['messageClass']
    except KeyError:
        print(f'Document {path} does not have a messageClass, ignoring...')
        return

    body = doc.get('bodyText')

    if doc_type == 'IPM.Note':
        sender = get_sender(doc)
        recipients = get_recipients(doc)

        for email_d in [sender] + recipients:
            addr, name, com_type = email_d.get(
                'addr'), email_d.get('name'), email_d.get('type')
            if not addr:
                continue
            org_info.email_addrs_w_names.setdefault(addr, set())
            if name:
                org_info.email_addrs_w_names[addr].add(name)

            if not user:
                continue

            if check_if_folder_is_sent(path):
                greeting = get_greeting(body) if body else None
                target_com_types = ('to', 'cc', 'bcc', 'recipient')
                if '@' in addr:
                    org_info.org_domains.add(addr.split('@')[1])
            else:
                greeting = None
                target_com_types = ('sender',)
                if user and name == user.name and '@' in addr:
                    org_info.org_domains.add(addr.split('@')[1])

            if com_type in target_com_types:
                user.communicators_counter[addr] += 1
                user.communicators.setdefault(
                    addr, {'names': set(), 'user_greetings': Counter(), 'com_types': Counter()})
                user.communicators[addr]['com_types'][com_type] += 1  # type: ignore # nopep8
                if name:
                    user.communicators[addr]['names'].add(name)  # type: ignore
                if greeting and com_type == 'to':
                    user.communicators[addr]['user_greetings'][greeting] += 1  # type: ignore # nopep8

    elif doc_type == 'IPM.Contact' and user:
        if contact := get_contact(doc):
            user.contacts.append(contact)
            email_addr = contact['email_addr']
            org_info.email_addrs_w_names.setdefault(email_addr, set())
            if name := contact.get('name'):
                org_info.email_addrs_w_names[email_addr].add(name)

    if body:
        email_addrs = find_email_addresses(body)
        for addr in email_addrs:
            org_info.email_addrs_w_names.setdefault(
                addr,  set())  # type: ignore
        org_info.urls.update(find_urls(body))
        phone_nums_w_context = find_phone_nums(body, context=True)
        for match in phone_nums_w_context:
            context1, phone_num, context2 = match
            org_info.phone_nums_w_context.setdefault(
                phone_num, set())
            org_info.phone_nums_w_context[phone_num].add(
                f'{context1} [PHONE NUM] {context2}')


def get_contact_info(doc_ref: DocRef) -> OrgInfo:
    org_info = OrgInfo()
    for user_folder in doc_ref.get_user_folders():
        process_user(user_folder, org_info, doc_ref)

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


def save_signatures_from_all_docs(freq_text_blocks: list[str], output: Path) -> None:
    signatures = []
    for text_block in freq_text_blocks:
        if find_email_addresses(text_block) and find_phone_nums(text_block):
            signatures.append(text_block)
    dump_json(signatures, output)


def save_contact_info(org_info: OrgInfo, output: Path) -> None:
    user_folder = output / 'users'
    email_addrs_folder = output / 'email_addrs'
    urls_folder = output / 'urls'
    phone_nums_folder = output / 'phone_nums'
    for folder in (user_folder, email_addrs_folder, urls_folder, phone_nums_folder):
        folder.mkdir(parents=True, exist_ok=True)

    for user in org_info.users:

        communicators_to_save = {}
        for addr, count in user.communicators_counter.most_common()[:20]:
            communicators_to_save[addr] = user.communicators[addr]
            communicators_to_save[addr]['count'] = count
            communicators_to_save[addr]['names'].update(
                org_info.email_addrs_w_names[addr])
            communicators_to_save[addr]['user_greetings'] = [
                greeting for greeting, count in communicators_to_save[addr]['user_greetings'].most_common(3)]

        user_dict = user.__dict__
        user_dict.pop('communicators_counter')
        user_dict['communicators'] = communicators_to_save

        dump_json(user.__dict__, user_folder / f'{user.name}.json')

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


def main(doc_ref: DocRef, output: Path) -> None:
    """Gets emails, urls, phone numbers, contacts, etc. from PST docs.mmary_

    Args:
        doc_ref (DocRef): Document reference.
        output (Path): Path for output fulder.
    """
    org_info = get_contact_info(doc_ref)
    save_contact_info(org_info, output)
