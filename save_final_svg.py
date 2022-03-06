import os
import shutil
from pathlib import Path

file_list = []
for p in Path('experiment/').iterdir():
    for s in p.rglob('iter_249.svg'):
        file_list.append(str(s))
        
for source_path in file_list:
    char_class = source_path[11]
    new_path = f'final_svg/{char_class}.svg'
    shutil.copy(source_path, new_path)
