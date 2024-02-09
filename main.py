"""
Analyzes emails and other PST docs and saves results in output folder.
"""

import argparse
from collections import Counter
import os

import tldextract
from tqdm import tqdm

from get_entities import main as get_entities_main
from utils import find_email_addresses, find_urls, load_json, dump_json


class User():
    """
    Class for users (employees, etc.).
    """

    def __init__(self, folder, name):
        self.name = name
        self.folder = folder
        self.contacts = {}
        self.communicators = Counter()

    def get_contact(self, contactItem):
        """Gets 1st email and display name."""
        name = contactItem.get('email1DisplayName')
        if not name:
            name = contactItem.get('givenName', '') + \
                contactItem.get('surname', '')

        if email := contactItem.get('email1EmailAddress'):
            self.contacts.setdefault(email, {'names': set()})
            if name:
                self.contacts[email]['names'].add(name)

    def get_top_communicators(self, n):
        return self.communicators.most_common(n)


class Org():
    """
    Class for processing organizations' PST docs (emails, etc.).
    """

    def __init__(self, folder, output, save=True):
        self.folder = folder
        self.output = output
        self.name = os.path.basename(os.path.normpath(folder))
        self.users = []
        self.email_addrs = {}
        self.email_domains = Counter()
        self.email_domains_to_addrs = {}
        self.urls = set()
        self.url_domains = Counter()
        self.url_domains_to_urls = {}
        self.save = save

    def get_user_email_and_url_info(self):
        for user_folder in os.listdir(self.folder):
            user_folder = os.path.join(self.folder, user_folder)
            self.process_user(user_folder)
        self.get_email_domains()
        self.get_url_domains()
        if self.save:
            for folder in ('users', 'email_addrs', 'urls'):
                os.makedirs(os.path.join(self.output, folder), exist_ok=True)
            self.save_func()

    def process_user(self, user_folder):
        user_name = os.path.basename(os.path.normpath(user_folder))
        user = User(user_folder,  user_name)
        self.users.append(user)
        for root, dir, files in tqdm(list(os.walk(user_folder)), f'Processing {user_name}...'):
            for f in files:
                self.process_file(os.path.join(root, f), user)

    def process_file(self, path, user):
        item = load_json(path)
        type_ = item['messageClass']

        if type_ == 'IPM.Note':
            sender = get_sender(item)
            recipients = get_recipients(item)
            for name, addr in set([sender]) | recipients:
                self.email_addrs.setdefault(addr, {'names': set()})
                self.email_addrs[addr]['names'].add(name)
            # Checks if this is a sent item. Might need more robust method.
            if 'Sent Items' in os.path.normpath(path).split(os.pathsep):
                user.communicators.update([r[1] for r in recipients])
            else:
                user.communicators[sender[1]] += 1

        elif type_ == 'IPM.Contact':
            user.get_contact(item)

        if body := item.get('bodyText'):
            email_addrs = find_email_addresses(body)
            for addr in email_addrs:
                self.email_addrs.setdefault(addr, {'names': set()})
            self.urls.update(find_urls(body))

    def get_email_domains(self):
        for address in self.email_addrs:
            if address and '@' in address:
                domain = address.split('@')[1]
                self.email_domains[domain] += 1
                self.email_domains_to_addrs.setdefault(domain, [])
                self.email_domains_to_addrs[domain].append(address)
        self.email_domains = self.email_domains.most_common()

    def get_url_domains(self):
        for url in self.urls:
            parsed = tldextract.extract(url)
            domain = parsed.domain + '.' + parsed.suffix
            if domain:
                self.url_domains[domain] += 1
                self.url_domains_to_urls.setdefault(domain, [])
                self.url_domains_to_urls[domain].append(url)
        self.url_domains = self.url_domains.most_common()

    def get_vendors(self):
        print('Getting keywords, vendors...')
        get_entities_main(self.folder, os.path.join(
            self.output, 'vendors'))

    def save_func(self):
        # User info
        for user in self.users:
            dump_json(user.contacts, os.path.join(
                self.output, 'users', f'{user.name}_contacts.json'), default=list)
            dump_json(dict(user.communicators.most_common(20)), os.path.join(
                self.output, 'users', f'{user.name}_top_communicatiors.json'))
        # Email addrs
        dump_json(self.email_addrs, os.path.join(
            self.output, 'email_addrs',   'all_emails.json'), default=list)
        for i, (domain, count) in enumerate(self.email_domains[:10]):
            dump_json(self.email_domains_to_addrs[domain], os.path.join(
                self.output, 'email_addrs', f'{i}_emails_for_{domain}.json'))
        # URLs
        dump_json(self.urls, os.path.join(
            self.output, 'urls',   'all_urls.json'), default=list)
        dump_json(dict(self.url_domains), os.path.join(
            self.output, 'urls',   'all_domains.json'))
        for i, (domain, count) in enumerate(self.url_domains[:10]):
            dump_json(self.url_domains_to_urls[domain], os.path.join(
                self.output, 'urls', f'{i}_urls_for_{domain}.json'))


def get_sender(email):
    name = email.get('senderName', '')
    address = email.get('senderEmailAddress', '').lower()
    if not address:
        if match := find_email_addresses(name):
            address = match[0].lower()
    return name, address


def get_recipients(email):
    r_names, r_addrs = [], []
    for k in ('displayTo', 'displayCC'):
        if v := email.get(k):
            r_names.extend([name.strip() for name in v.split(';')])
    for r in email.get('recipients', []):
        r_addrs.append(r['smtpAddress'].lower())
    r_names.extend([None for x in range(len(r_addrs) - len(r_names))])
    recipients = set([(name, addr)
                      for name, addr in zip(r_names, r_addrs)])
    return recipients


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Analyzes emails and other PST docs and saves results in output folder.')

    parser.add_argument('input_folder')
    parser.add_argument('output_folder')
    args = parser.parse_args()
    org = Org(args.input_folder, args.output_folder)
    org.get_user_email_and_url_info()
    org.get_vendors()
