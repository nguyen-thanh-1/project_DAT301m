import json
import io

with open('C:/Users/Admin/Desktop/Project_DAT301m/models/face_recognition/siamese/train_siamese.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

with io.open('C:/Users/Admin/Desktop/Project_DAT301m/models/face_recognition/siamese/extracted_cells.txt', 'w', encoding='utf-8') as out:
    for cell in nb['cells']:
        if cell['cell_type'] == 'code':
            out.write("".join(cell['source']))
            out.write("\n---\n")
