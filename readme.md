# PST Analysis

## Setup

1. Install python dependencies (`pip install -r requirements.txt`).
2. Install node dependencies (go to *preprocess/process_pst_js/* and run `npm install`)
3. Comment out lines 162-165 in *preprocess/process_pst_js/node_modules/pst-extractor/dist/PSTFolder.class.js*. Not sure if this is a bug or I'm doing something wrong, but it only seems to extract all the PST items if these lines are commented out: 

  ```
  if ((emailRow && emailRow.itemIndex == -1) || !emailRow) {
    // no more!
    return null;
  }
  ```
4. Install and run Gophish (https://getgophish.com/)
5. Get Gophish api key and add to *config/phish_config.json*.

## Usage

**Note:** See *project.py* documentation for parameters, etc. The only thing that needs to be specified is a source and project folder.

1. Specify a source (path of folder containing PST files or email files, or the path of a single PST file.) and project (output) folder in the *config/project.config* file.

    *Note:* Each PST file or subfolder should correspond to an individual's communications (for getting things like their top communicators, etc.) If the program can't find an "owner" (by inferring from sent folders or the `PR_RECEIVED_BY_NAME` Outlook API property) it will just get non-user-specific data from the folder/PST. 

2. Run the first cell in *main.ipynb* to initialize a *Project* and run the analysis. The results will be in *[project-folder]/output*. You can initialze a *Project* instance in the same folder and already-completed items won't be run again.

3. You can try out the semantic search in last cell of *main.ipynb*.

4. Set up a phishing campaign with *phish.ipynb*. The input to a campaign should be a user JSON file produced by a *Project* in *[project-folder]/output/contact_info/users*. It will set up a campaign to send emails from the user's address to his/her top communicators. The sender name and email address are generated automatically but can be manually specified in the config (e.g. if there's no SMTP-type email address list for the sender in the user file).

## Output

```
Project folder
 ┣ _util : Util folder
 ┣ docs : Extracted/processed PST files, email files, etc.
 ┗ output
 ┃ ┣ contact_info
 ┃ ┃ ┣ email_addrs : company/org emails and all emails
 ┃ ┃ ┣ phone_nums : phone nums w/ context
 ┃ ┃ ┣ urls : company/org emails and all emails
 ┃ ┃ ┣ users : files for users containing their contacts, top communicators, job title(s), signature(s) etc.
 ┃ ┗ entities
 ┃ ┃ ┣ vendors_default.json: Vendor names JSON
 ┃ ┃ ┗ vendors_default.md: Vendor names markdown
```

**Notes:**
 - **user job titles/signatures**: Signatures for users are identified from frequent text blocks that have their name, a phone number and/or email address. Job titles are extracted from the signature using a list of 200k job titles. Multiple candidates are included in case of false positives.

     The next step would be classifying the job titles as executive/not, which shouldn't be difficult.
