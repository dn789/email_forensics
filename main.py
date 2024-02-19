"""
Analyzes emails and other PST docs and saves results in output folder.
"""

import argparse
from collections import Counter
import os

import tldextract
from tqdm import tqdm

from preproccess import main as preprocess_main
from get_entities import main as get_entities_main
from utils import find_email_addresses, find_phone_nums, find_urls, load_json, dump_json, get_sender, get_recipients


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
            self.contacts.setdefault(email, {})
            if name:
                self.contacts[email]['name'] = name

    def get_top_communicators(self, n):
        return self.communicators.most_common(n)


class Org():
    """
    Class for processing organizations' PST docs (emails, etc.).
    """

    def __init__(self, folder, output, config={}):
        self.folder = folder
        if not os.path.isdir(folder):
            raise FileNotFoundError('Input folder not found.')
        self.name = os.path.basename(os.path.normpath(folder))
        self.output = output
        os.makedirs(self.output, exist_ok=True)
        self.util_folder = os.path.join(output, '_util')
        os.makedirs(self.util_folder, exist_ok=True)
        self.misc_folder = os.path.join(output, 'misc')
        os.makedirs(self.misc, exist_ok=True)

        self.log_path = os.path.join(
            self.util_folder, 'log.json')
        if not os.path.isfile(self.log_path):
            dump_json({}, self.log_path)
            self.log = {}
        else:
            self.log = load_json(self.log_path)

        self.users = []
        self.email_addrs = {}
        self.email_domains = Counter()
        self.email_domains_to_addrs = {}
        self.urls = set()
        self.url_domains = Counter()
        self.url_domains_to_urls = {}
        self.org_domains = set()

        unique_body_paths_f = os.path.join(
            self.util_folder, 'unique_body_paths.json')
        if not os.path.isfile(unique_body_paths_f):
            self.unique_body_paths = set()
            body_texts = set()
            for root, dirs, files in os.walk(self.folder):
                for f in files:
                    path = os.path.join(root, f)
                    body = load_json(path).get('bodyText', '')
                    if body and body not in body_texts:
                        self.unique_body_paths.add(path)
            dump_json(self.unique_body_paths, os.path.join(
                self.util_folder, 'unique_body_paths.json'), default=list)
        else:
            self.unique_body_paths = load_json(unique_body_paths_f)
        if config:
            if config.get('get_user_email_and_url_info', {}).get('run'):
                self.get_user_email_and_url_info()
            if config.get('preprocess_emails', {}).get('run'):
                self.preprocess_emails(
                    **config.get('preprocess_emails', {}).get('kwargs', {}))
            if config.get('get_vendors', {}).get('run'):
                self.get_vendors(
                    **config.get('get_vendors', {}).get('kwargs', {}))

    def get_user_email_and_url_info(self):
        for user_folder in os.listdir(self.folder):
            user_folder = os.path.join(self.folder, user_folder)
            self.process_user(user_folder)
        self.get_email_domains()
        self.get_url_domains()
        self.get_org_emails_and_urls()
        # Saves to output
        for folder in ('users', 'email_addrs', 'urls'):
            os.makedirs(os.path.join(self.output, folder), exist_ok=True)
         # User info
        for user in self.users:
            dump_json(user.contacts, os.path.join(
                self.output, 'users', f'{user.name}_contacts.json'), default=list)
            dump_json(dict(user.communicators.most_common(20)), os.path.join(
                self.output, 'users', f'{user.name}_top_communicatiors.json'))
        # Email addrs
        dump_json(self.email_addrs, os.path.join(
            self.output, 'email_addrs',   'all_emails.json'), default=list)
        dump_json(self.org_email_addrs, os.path.join(
            self.output, 'email_addrs',   'org_emails.json'), default=list)
        # URLs
        dump_json(self.urls, os.path.join(
            self.output, 'urls',   'all_urls.json'), default=list)
        dump_json(self.org_urls, os.path.join(
            self.output, 'urls',   'org_urls.json'), default=list)
        self.log['get_user_email_and_url_info'] = True
        dump_json(self.log, self.log_path)

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
            for name, addr in [sender] + recipients:
                if addr:
                    self.email_addrs.setdefault(addr, {'names': set()})
                    if name:
                        self.email_addrs[addr]['names'].add(name)
            # Checks if this is a sent item. Might need more robust method.
            if 'Sent Items' in os.path.normpath(path).split(os.path.sep):
                user.communicators.update([r[1] for r in recipients])
                # Get company/organization domains from sender in sent items.
                if '@' in sender[1]:
                    self.org_domains.add(sender[1].split('@')[1])

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

    def get_org_emails_and_urls(self):
        self.org_email_addrs = set()
        self.org_urls = set()
        user_names = [user.name for user in self.users]
        for name in user_names:
            for addr, v in self.email_addrs.items():
                if name in v['names'] and '@' in addr:
                    self.org_domains.add(addr.split('@')[1])
        for domain in self.org_domains:
            if domain not in ["hotmail.com", "gmail.com", "yahoo.com"]:
                self.org_email_addrs.update(
                    self.email_domains_to_addrs[domain])
                self.org_urls.update(self.url_domains_to_urls[domain])

    def get_url_domains(self):
        for url in self.urls:
            parsed = tldextract.extract(url)
            domain = parsed.domain + '.' + parsed.suffix
            if domain:
                self.url_domains[domain] += 1
                self.url_domains_to_urls.setdefault(domain, [])
                self.url_domains_to_urls[domain].append(url)
        self.url_domains = self.url_domains.most_common()

    def preprocess_emails(self, **kwargs):
        print('Preprocessing emails...')
        preprocess_main(self.unique_body_paths, self.util_folder, **kwargs)
        self.log['preprocess_emails'] = True
        dump_json(self.log, self.log_path)
        # Uses frequently occuring strings to find signatures
        signatures = set()
        for match in load_json(os.path.join(self.util_folder, 'text_to_remove.json')):
            if find_email_addresses(match) and find_phone_nums(match):
                signatures.add(match)
        dump_json(signatures, os.path.join(
            self.misc_folder, 'signatures.json'), default=list)

    def get_vendors(self, **kwargs):
        if not self.log.get('preprocess_emails'):
            self.preprocess_emails()
        print('Getting keywords, vendors...')
        vendors_folder = os.path.join(self.output, 'vendors')
        os.makedirs(vendors_folder, exist_ok=True)
        get_entities_main(self.unique_body_paths,
                          vendors_folder, self.util_folder, **kwargs)

    def search_docs(self, search_string):
        pass


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Analyzes emails and other PST docs and saves results in output folder.')

    parser.add_argument('--input', default=None)
    parser.add_argument('--output', default=None)
    parser.add_argument('--config', default=None,
                        help='Run with input, output, parameters from config file.')
    args = parser.parse_args()

    if args.input and args.output:
        org = Org(args.input, args.output)
        org.get_user_email_and_url_info()
        org.preprocess_emails()
        org.get_vendors()
    elif args.config:
        config = load_json(args.config)
        org = Org(config['input'], config['output'], config)
    else:
        raise ValueError(
            'Must provide (input  and output) or path to congfig file.')
