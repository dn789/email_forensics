from collections import Counter
import json
import os
import re


def get_addresses(email):
    addresses = Counter()

    for k, v in email.items():
        if k in ('From', 'To'):
            names_addresses = []
            for name_address_str in v.split(','):
                name_address_str = name_address_str.strip()
                match = re.match(r'"*([^"]+)"*\s*<(\S+)>|<*(\S+)>*',
                                 name_address_str)
                if not match:
                    continue
                if match.group(3):
                    address = match.group(3)
                    names_addresses.append(
                        {'name': None, 'address': address})
                    addresses[address] += 1
                else:
                    address = match.group(2)
                    names_addresses.append(
                        {'name': match.group(1).strip(), 'address': address})
                    addresses[address] += 1

            if k == 'From':
                names_addresses = names_addresses[0]
            email[k] = names_addresses
        addresses.update(re.findall(
            r'[a-zA-z0-9\.\_\-]+@[a-zA-z0-9\.\_\-]+\.[a-zA-z0-9]+', v))
    email['all_addresses'] = addresses


def process_email(email):
    get_addresses(email)
    return email


def process_folder(folder, save=True, output=None):
    """
    Analyzes emails in folder; formats email JSONs. 

    Parameters
    ----------
    folder : str
    save : bool, optional
        Save processed emails.
    output : _type_, optional
        Output folder, overwrites if None. 
    """
    output = output or folder
    addresses = Counter()
    for f in os.listdir(folder):
        email = json.load(open(os.path.join(folder, f)))
        email = process_email(email)
        addresses.update(email['addresses'])
        if save:
            json.dump(email, open(output, 'w'))

    # Extracts company addresses
    domains = Counter()
    domains_to_names = {}
    for address in addresses:
        name, domain = address.split('@')
        domains[domain] += 1
        domains_to_names.setdefault(domain, [])
        domains_to_names[domain].append(name)
    domains = domains.most_common(10)
