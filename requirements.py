import platform
import subprocess

from pathlib import Path


def check_if_target_code_is_commented(lines: list[str]) -> bool:
    if all(line.strip().startswith('//') for line in lines[161:165]):
        return True
    elif lines[160].strip() == '/*' and lines[165].strip() == '*/':
        return True
    return False


def fix_pst_extractor_code() -> bool:
    target_filepath = Path(
        'src/preprocess/process_pst_js/node_modules/pst-extractor/dist/PSTFolder.class.js')
    lines = target_filepath.read_text().split('\n')
    if check_if_target_code_is_commented(lines):
        return True
    try:
        assert lines[161].strip().startswith(
            'if ((emailRow && emailRow.itemIndex == -1) || !emailRow) {')
        assert lines[162].strip().startswith(
            '// no more!')
        assert lines[163].strip().startswith(
            'return null;')
        assert lines[164].strip().startswith('}')
        for i in (161, 162, 163, 164):
            lines[i] = '// ' + lines[i]
        target_filepath.write_text('\n'.join(lines))
        return True
    except AssertionError:
        return False


# JS dependencies
subprocess.run(
    ['npm', 'install', '--prefix', './src/preprocess/process_pst_js/'], check=True)
PST_EXTRACTOR_CODE_FIXED = fix_pst_extractor_code()

# Python dependencies
subprocess.run(
    ['pip', 'install', '--no-deps', '-r', 'requirements.txt'], check=True)
PYTORCH_INSTALLED = False
if platform.system() == 'Linux':
    subprocess.run(
        ['pip', 'install', 'torch==2.3.0'], check=True)
    PYTORCH_INSTALLED = True

if PYTORCH_INSTALLED and PST_EXTRACTOR_CODE_FIXED:
    print('\n\nSuccessfully installed JS and Python dependencies.\n\n')
else:
    print('\n\nAlmost done installing JS and Python depdendencies:\n')
if not PYTORCH_INSTALLED:
    print('\n- You need to install PyTorch manually: https://pytorch.org/get-started/locally/')
if not PST_EXTRACTOR_CODE_FIXED:
    print(
        f"""\n- Could not edit pst_extractor code.\nThis is necessary for the program to run properly.\nSee step 4 of "Manually install JS and Python dependencies" in the readme.""")
