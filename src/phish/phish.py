from pathlib import Path
from typing import Any
from gophish import Gophish, api
from gophish.models import Group, User, Campaign, Page, SMTP, Template

from utils.doc import parse_name
from utils.io import load_json


def create_group_from_user(user: dict[str, Any], group_name: str) -> Group:
    targets = []
    for email_addr, d in user.get('communicators', {}).items():
        if '@' not in email_addr:
            continue
        target_name = {}
        for name in d['names']:
            parsed_name = parse_name(name)
            if parsed_name:
                target_name = parsed_name
                break
        else:
            if d['names']:
                target_name = {'first_name': d['names'][0]}
        target = User(first_name=target_name.get('first_name'),
                      last_name=target_name.get('last_name'), email=email_addr)
        targets.append(target)
    group = Group(name=group_name, targets=targets)
    return group


def create_campaign(user_file: Path, config: dict[str, Any], launch: bool = False, add_signature_to_template: bool = True) -> None:
    """Creates Gophish campaign from user_file

    Args:
        user_file (Path): user JSON file produced by a Project in
            [project-folder]/output/contact_info/users.
        config (dict[str, Any]): {
            "api_key" (str): Gophish API key,
            "sender_name" (str | None): Automatically generated; specify to overwrite. 
            "sender_email_addr" (str | None): Automatically generated; specify to overwrite.
            "SMTP" (dict): Args for SMTP model; need to specify "name" and "host".
            "Page" (dict): Args for Page model; need to specify "name" and "text_path".
            "Template" (dict): Args for Template model; need to specify "name" and "text_path"
        }
        launch (bool, optional): Whether to launch campaign. If False, just
            posts the individual models (Group, Page, SMTP, etc.). Defaults
            to False.
        add_signature_to_template (bool, optional): Whether to add signature
            from user_file to end of phishing template. Defaults to True.

    Raises:
        ValueError: Can't find an email address for the sender.
    """

    user_d = load_json(user_file)

    config.setdefault('SMTP', {})
    sender_name = config.get('sender_name') or user_d['name']
    if config_sender_email_addr := config.get('sender_email_addr'):
        sender_email_addr = config_sender_email_addr
    else:
        for addr in user_d['email_addrs']:
            if '@' in addr:
                sender_email_addr = addr
                break
            else:
                sender_email_addr = None
    if not sender_email_addr and not config['SMTP'].get('from_address'):
        raise ValueError(
            f'No email address found for user in {user_file} and none provided in config.')

    config['SMTP']['from_address'] = config['SMTP'].get(
        'from_address') or f'{sender_name} <{sender_email_addr}>'

    smtp = SMTP(**config['SMTP'])

    groups = []
    group = create_group_from_user(
        user_d, group_name=f'{user_file.stem} Group')
    groups.append(group)

    page_html = open(config['Page']['text_path']).read()
    config['Page']['html'] = page_html
    config['Page'].pop('text_path')
    page = Page(**config['Page'])

    template_text = open(config['Template']['text_path']).read()
    if add_signature_to_template:
        if sigs := user_d['signatures']:
            signature = sigs[0]
            template_text += f'\n{signature}'
    config['Template']['text'] = template_text
    config['Template'].pop('text_path')
    template = Template(**config['Template'])

    api_key = config['api_key']
    api = Gophish(api_key, verify=False)

    if launch:
        campaign = Campaign(groups=groups, name='Test Campaign',
                            template=template, page=page, smtp=smtp)
        api.campaigns.post(campaign)
    else:
        api.groups.post(group)
        api.pages.post(page)
        api.templates.post(template)
        api.smtp.post(smtp)
