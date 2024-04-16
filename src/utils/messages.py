# get_contact_info
MULTI_USER_PROMPT = """If all these items represent the same user, enter "all". 
Otherwise, list items to process as users, using parentheses to group items representing the same user, e.g:
    "1" to only process potential user 1
    "(1, 2), 3" to process 1 and 2 as one user, 3 separately

If more than one user is selected, contacts won't be collected for any of the users (because we can't
tell which user the contact belongs to).

"""
