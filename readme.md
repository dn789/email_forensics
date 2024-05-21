# Email Forensics

## Overview

This is a forensic analysis tool for emails. Currently in the alpha phase, the tool:

-   extracts basic info (email addresses, web domains, etc).
-   finds secrets (passwords, API keys, etc.)
-   gets sentence embeddings and provides a semantic search interface
-   uses named entity recognition and semantic search to get a list of vendor names:

    1. Uses semantic search to get emails relevant to terms like "invoice", "payment", "vendor" etc.
    2. Uses a named entity recognition tagger to get organization names in those emails
    3. Filters out org entities that occur frequently in entire set of emails.

-   sets up phishing campaigns using automatically extracted greetings and signatures.

The input should be a folder containing an organization's PST files/text email files, with each PST file/text file subfolder corresponding to an employee/member of that organization.

## Setup

Requires: Python 3.11, Node.js 20.12, Golang

1. Install JS and Python dependendencies: `python requirements.py`

    - If you're not using Linux, you'll need to install PyTorch with CUDA support manually.
    - The setup script will try to edit the code of a JS dependency. If it fails, you'll have to do it manually.
    - For the above issues or if the script fails, see manual installation instructions [below](#manually-install-js-and-python-dependencies).

2. Install and run `Gophish` (https://getgophish.com/). Get `Gophish` api key and add to `config/phish_config.json`.
3. Install `Gitleaks` (https://github.com/gitleaks) (Golang required)

## Usage

### Note

See `project.py` documentation for parameters, etc. **The only thing that needs to be specified in the config file is a source and project folder.**

**If you want to find passwords, API keys, etc., you also need to specify the `gitleaks_path` (path to the executable) in `find_secrets` in the config file. The `gitleaks_config` path is already specified.**

1. Specify a source (path of folder containing PST files or email files, or the path of a single PST file.) and project (output) folder in the `config/project_config.json` file.

    _Note:_ Each PST file or subfolder should correspond to an individual's communications (for getting things like their top communicators, etc.) If the program can't find an "owner" (by inferring from sent folders or the `PR_RECEIVED_BY_NAME` Outlook API property) it will just get non-user-specific data from the folder/PST.

2. Run the first cell in `main.ipynb` to initialize a `Project` and run the analysis. The results will be in `[project-folder]/output`. You can initialze a `Project` instance in the same folder and already-completed items won't be run again.

3. You can try out the semantic search in last cell of `main.ipynb`.

4. Set up a phishing campaign with `phish.ipynb`. The input to a campaign should be a user JSON file produced by a `Project` in `[project-folder]/output/contact_info/users`. It will set up a campaign to send emails from the user's address to his/her top communicators. The sender name and email address are generated automatically but can be manually specified in the config (e.g. if there's no SMTP-type email address list for the sender in the user file).

## Output

<pre>
<b>Project folder</b> 
 ┣ <b>_util</b> : Util folder
 ┣ <b>docs</b> : Extracted/processed PST files, email files, etc.
 ┗ <b>output</b>
 ┃ ┣ <b>contact_info</b>
 ┃ ┃ ┣ <b>email_addrs</b> : company/org emails and all emails
 ┃ ┃ ┣ <b>phone_nums</b> : phone nums w/ context
 ┃ ┃ ┣ <b>urls</b> : company/org emails and all emails
 ┃ ┃ ┣ <b>users</b> : files for users containing their contacts, top communicators, job title, signature, etc.
 ┃ ┗ <b>entities</b> : vendor names, etc.
 ┃ ┗ <b>secrets.json</b> : passwords, API keys, etc.
</pre>

**Notes:**

-   **user job titles/signatures**: Signatures for users are identified from frequent text blocks that have their name, a phone number and/or email address. Job titles are extracted from the signature using a list of 200k job titles. Multiple candidates are included in case of false positives.

-   **find_secrets/gitleaks**: Change the entropy of the rules in the gitleaks config file (`config/gitleaks.toml`) to increase or decrease the sensitivity of the search. The most productive rule should be "generic-api-key".

## Manually install JS and Python dependencies

1. Install python dependencies (`pip install --no-deps -r requirements.txt`). **Make sure --no-deps flag is used.**
2. Install PyTorch 2.2.2>=2.3.0 with CUDA support (https://pytorch.org/get-started/locally/)
3. Install node dependencies (go to `preprocess/process_pst_js/` and run `npm install`)
4. Comment out lines 162-165 in `preprocess/process_pst_js/node_modules/pst-extractor/dist/PSTFolder.class.js`. Not sure if this is a bug, but it only seems to extract all the PST items if these lines are commented out:

```
if ((emailRow && emailRow.itemIndex == -1) || !emailRow) {
  // no more!
  return null;
}
```
