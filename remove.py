import os
import glob


def removeFiles():
    # remove old executable files
    files_exe = glob.glob('instance/downloads/*')
    for f in files_exe:
        os.remove(f)

    # remove old CVS files
    files_csv = glob.glob('instance/testCSV/*')
    for f_cs in files_csv:
        os.remove(f_cs)

    # remove old json files
    files_json = glob.glob('instance/result/*')
    for f_js in files_json:
        os.remove(f_js)

    print("Previous session files removed")
