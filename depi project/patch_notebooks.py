import json
import os

def patch_feature_eng():
    path = "Feature_Engineering_Artworks.ipynb"
    if not os.path.exists(path):
        print(f"{path} not found")
        return
        
    with open(path, 'r', encoding='utf-8') as f:
        nb = json.load(f)
        
    for cell in nb.get('cells', []):
        if cell.get('cell_type') == 'code':
            source = cell.get('source', [])
            new_source = []
            for line in source:
                if line.startswith("conda ") or line.startswith("pip ") or line.startswith("git "):
                    new_source.append("!" + line)
                elif line.startswith("cd "):
                    new_source.append("%" + line)
                else:
                    new_source.append(line)
            cell['source'] = new_source
            
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=2, ensure_ascii=False)
    print(f"Patched {path}")

def patch_pipeline():
    path = "Full_EmotionArt_Pipeline (1)20.ipynb"
    if not os.path.exists(path):
        print(f"{path} not found")
        return
        
    with open(path, 'r', encoding='utf-8') as f:
        nb = json.load(f)
        
    for cell in nb.get('cells', []):
        if cell.get('cell_type') == 'code':
            source = cell.get('source', [])
            new_source = []
            for line in source:
                # Fix REPO_DIR
                if 'REPO_DIR = r"D:\Projects\stylegan2-ada-pytorch"' in line:
                    line = line.replace('REPO_DIR = r"D:\Projects\stylegan2-ada-pytorch"', 'REPO_DIR = "stylegan2-ada-pytorch"')
                if 'REPO_DIR = r"D:\\Projects\\stylegan2-ada-pytorch"' in line:
                    line = line.replace('REPO_DIR = r"D:\\Projects\\stylegan2-ada-pytorch"', 'REPO_DIR = "stylegan2-ada-pytorch"')
                if 'REPO_DIR = "D:\\Projects\\stylegan2-ada-pytorch"' in line:
                    line = line.replace('REPO_DIR = "D:\\Projects\\stylegan2-ada-pytorch"', 'REPO_DIR = "stylegan2-ada-pytorch"')
                
                # Fix train.py path
                if '!python /content/stylegan2-ada-pytorch/train.py' in line:
                    line = line.replace('!python /content/stylegan2-ada-pytorch/train.py', '!python stylegan2-ada-pytorch/train.py')
                    
                # Fix torch.BytesIO
                if 'torch.BytesIO' in line:
                    line = line.replace('torch.BytesIO', 'io.BytesIO')
                
                new_source.append(line)
            cell['source'] = new_source
            
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=2, ensure_ascii=False)
    print(f"Patched {path}")

if __name__ == "__main__":
    patch_feature_eng()
    patch_pipeline()
