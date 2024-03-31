/* 
Extract messages, etc. from PST files

IMPORTANT: Comment out node_modules/pst-extractor/dist/PSTFolder.class.js:162-165:

  f ((emailRow && emailRow.itemIndex == -1) || !emailRow) {
    // no more!
    return null;
  }

Command line:
  node process_pst.js <input pst or folder> <output folder>
*/

import path from "path";
import { convert } from "html-to-text";
import { PSTFile } from "pst-extractor";
import { mkdirSync, writeFileSync } from "fs";

// Remove keys from item JSON
const removeKeys = [
    "descriptorIndexNode",
    "localDescriptorItems",
    "pstTableBC",
    "pstTableItems",
    "pstFile",
    "recipientTable",
    "attachmentTable",
    "conversationId",
];

function processPST(input, output) {
    let pstFile;
    try {
        pstFile = new PSTFile(input);
    } catch (err) {
        console.log("Error opening: " + input);
    }
    mkdirSync(output, { recursive: true });
    processFolder(pstFile.getRootFolder(), output);
}

function processFolder(folder, output) {
    if (folder.contentCount > 0) {
        let count = folder.contentCount;
        let i = 0;

        while (count > 0) {
            i++;
            let item = folder.getNextChild();
            if (item) {
                saveItem(item, output, i);
            }
            count -= 1;
        }
    }
    if (folder.hasSubfolders) {
        let childFolders = folder.getSubFolders();
        for (let childFolder of childFolders) {
            let currOutput = path.join(output, childFolder.displayName);
            mkdirSync(currOutput, { recursive: true });
            processFolder(childFolder, currOutput);
        }
    }
}

function saveItem(item, output, i) {
    let itemObj = item.toJSON();
    itemObj = Object.fromEntries(
        Object.entries(itemObj).filter(([_, v]) => !["", null].includes(v))
    );
    let recipients = [];
    let to = itemObj.displayTo;
    let cc = itemObj.displayCC;
    let bcc = itemObj.displayBCC;

    [...Array(item.numberOfRecipients).keys()].forEach((r) => {
        let recipient = item.getRecipient(r);
        let recipientType;
        let emailAddress = recipient.emailAddress;

        if (
            to &&
            to.includes(recipient.displayName) |
                to.includes(recipient.emailAddress)
        ) {
            recipientType = "to";
        } else if (
            cc &&
            cc.includes(recipient.displayName) |
                cc.includes(recipient.emailAddress)
        ) {
            recipientType = "cc";
        } else if (
            bcc &&
            bcc.includes(recipient.displayName) |
                bcc.includes(recipient.emailAddress)
        ) {
            recipientType = "bcc";
        }
        recipients.push({
            name: recipient.displayName,
            addr: emailAddress,
            type: recipientType,
        });
    });

    itemObj.recipients = recipients;

    removeKeys.forEach((k) => delete itemObj[k]);

    try {
        if (item.body) {
            itemObj.bodyText = item.body;
        } else if (item.bodyRTF) {
            itemObj.bodyText = item.bodyRTF;
        }
        if (item.bodyHTML) {
            itemObj.bodyHTML = item.bodyHTML;
            if (!itemObj.bodyText) {
                itemObj.bodyText = htmlToText(item.bodyHTML);
            }
        }
    } catch (err) {}

    let type = item.messageClass.slice(4);
    let itemString = JSON.stringify(itemObj);
    output = path.join(output, i.toString() + "_" + type + ".json");
    writeFileSync(output, itemString);
}

function htmlToText(HTML) {
    return convert(HTML, {
        selectors: [
            { selector: "a", options: { ignoreHref: true } },
            { selector: "img", format: "skip" },
        ],
    });
}

// Command line
let input = process.argv[2];
let output = process.argv[3];

processPST(input, output);
