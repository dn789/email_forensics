""" 

1) pip install Aspose.Email-for-Python-via-NET
2) specify file path

"""
from aspose.email.storage.pst import PersonalStorage

FILEPATH = None

personalStorage = PersonalStorage.from_file(FILEPATH)

# Get folders' collection
folderInfoCollection = personalStorage.root_folder.get_sub_folders()

# Extract folders' information
for folderInfo in folderInfoCollection:
    print("Folder: " + folderInfo.display_name)
    print("Total Items: " + str(folderInfo.content_count))
    print("Total Unread Items: " + str(folderInfo.content_unread_count))
