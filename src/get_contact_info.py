"""Gets emails, urls, phone numbers, contacts, etc. from PST docs."""

from collections import Counter
from dataclasses import dataclass, field
from json import JSONDecodeError
from pathlib import Path
from typing import Callable, Any
import random
import re
import warnings

from fuzzysearch import find_near_matches
import tldextract
from tqdm import tqdm

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


JOB_TITLES = open('../data/wordlists/job_titles.txt').read().split('\n')


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


def get_greetings(text: str) -> list[str]:
    first_block = text.strip().split('\n')[0]
    greetings = [x for x in (first_block.split(',')[0].strip(),
                             first_block.split(':')[0].strip()) if x]
    return greetings


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


def get_job_titles(text: str) -> list[str]:
    return sorted([job for job in JOB_TITLES if re.findall(fr'\b{job}\b', text, flags=re.IGNORECASE)], key=len, reverse=True)[:2]


def get_signatures_for_user(user: UserInfo, paths: list[Path], n_docs_to_sample: int = 100) -> None:
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
        job_titles = get_job_titles(signatures[0])
        user.job_titles = job_titles


def determine_user(folder: Path, from_pst: bool = True) -> dict[str, Counter]:
    if from_pst:
        receving_names, receving_email_addrs = Counter(), Counter()
        for path in folder.rglob('*'):
            if path.is_file() and path.name.endswith('Note.json'):
                try:
                    doc = load_json(path)
                except JSONDecodeError:
                    print(path)
                    raise Exception
                if r_name := doc.get('receivedByName'):
                    receving_names[r_name] += 1
                if r_addr := doc.get('receivedByAddress'):
                    receving_email_addrs[r_addr] += 1
        if r_addrs_count := len(receving_email_addrs) > 1:
            warnings.warn(
                f"""\n{r_addrs_count} email addresses found for user in {folder.name}:
                    {', '.join(receving_email_addrs)}
                    Make sure these belong to the same person.""")
        if r_names_count := len(receving_names) > 1:
            warnings.warn(
                f"""\n{r_names_count} names found for user in {folder.name}:
                    {', '.join(receving_names)}
                    Make sure these belong to the same person.""")
        return {'names': receving_names, 'addrs': receving_email_addrs}
    else:
        sender_names, sender_email_addrs = Counter(), Counter()
        for path in folder.rglob('*'):
            if not check_if_folder_is_sent(path):
                continue
            if path.is_file() and path.suffix == '.json':
                try:
                    doc = load_json(path)
                except JSONDecodeError:
                    print(path)
                    raise Exception
                sender = get_sender(doc)
                if r_name := sender.get('name'):
                    sender_names[r_name] += 1
                if r_addr := sender.get('addr'):
                    sender_email_addrs[r_addr] += 1
        if r_addrs_count := len(sender_email_addrs) > 1:
            warnings.warn(
                f"""\n{r_addrs_count} email addresses found for user in {folder.name}:
                    {', '.join(sender_email_addrs)}
                    Make sure these belong to the same person.""")
        if r_names_count := len(sender_names) > 1:
            warnings.warn(
                f"""\n{r_names_count} names found for user in {folder.name}:
                    {', '.join(sender_names)}
                    Make sure these belong to the same person.""")
        return {'names': sender_names, 'addrs': sender_email_addrs}


def process_user(user_folder: Path, org_info: OrgInfo, doc_ref: DocRef) -> None:
    from_pst = doc_ref.source_dict[user_folder.__str__()] == 'pst'
    user_d = determine_user(user_folder, from_pst=from_pst)
    names = set()
    for i, (name, count) in enumerate(user_d['names'].most_common()):
        if not i:
            rep_user_name = name
        names.add(name)
    if names:
        print(
            f'\nProcessing {user_folder.name}. (User: {rep_user_name})...')
        user = UserInfo(name=rep_user_name)
        user.all_names = names
        user.email_addrs.update(user_d['addrs'].keys())
        org_info.users.append(user)

        for path in user_folder.rglob('*'):
            if path.is_file():
                process_doc(path, user, org_info)

        print(f'\nGetting signatures...')
        get_signatures_for_user(
            user, doc_ref.get_paths_by_user_folder(user_folder, sent_only=True))
    else:
        warnings.warn(
            f'\nCan\'t find a user for {user_folder.name}. Processing without getting user-specific info... ')
        for path in user_folder.rglob('*'):
            if path.is_file():
                process_doc(path, None, org_info)


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
                greetings = get_greetings(body) if body else None
                target_com_types = ('to', 'cc', 'bcc', 'recipient')
                if '@' in addr:
                    org_info.org_domains.add(addr.split('@')[1])
            else:
                greetings = None
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
                if greetings and com_type == 'to':
                    user.communicators[addr]['user_greetings'].update(
                        greetings)

    elif doc_type == 'IPM.Contact' and user:
        if contact := get_contact(doc):
            user.contacts.append(contact)
            email_addr = contact['email_addr']
            org_info.email_addrs_w_names.setdefault(email_addr, set())
            org_info.email_addrs_w_names[email_addr].add(contact['name'])

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
    org_info = get_contact_info(doc_ref)
    save_contact_info(org_info, output)
