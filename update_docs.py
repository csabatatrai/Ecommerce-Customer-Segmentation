import os
import subprocess
import shutil
import re

# --- Konfiguráció ---
NOTEBOOKS_DIR = '.'
DOCS_DIR = 'docs'
IMAGES_DIR = os.path.join(DOCS_DIR, 'images')

def update_documentation():
    """
    A szkript célja, hogy a Jupyter notebookokat Markdown-ná konvertálva
    frissítse a docs mappát és a notebookokhoz tartozó képek hivatkozásait a .md fájlokban.
    """
    # Létrehozzuk a szükséges mappákat, ha nincsenek
    os.makedirs(IMAGES_DIR, exist_ok=True)
    
    # Csak a releváns notebookokat szedjük össze
    notebooks = [f for f in os.listdir(NOTEBOOKS_DIR) if f.endswith('.ipynb')]
    notebooks.sort()
    
    for notebook in notebooks:
        print(f"[{notebook}] Konvertálás Markdown-ná...")
        base_name = notebook.replace('.ipynb', '')
        md_file = os.path.join(DOCS_DIR, f"{base_name}.md")
        files_dir = os.path.join(DOCS_DIR, f"{base_name}_files")
        
        # 1. Notebook konvertálása a háttérben
        subprocess.run([
            "jupyter", "nbconvert", 
            "--to", "markdown", 
            notebook, 
            "--output-dir", DOCS_DIR
        ], check=True)
        
        # 2. Képek áthelyezése az 'images' mappába és átnevezésük névütközés ellen
        if os.path.exists(files_dir):
            for img_file in os.listdir(files_dir):
                src = os.path.join(files_dir, img_file)
                # Új név: pl. 01_data_preparation_output_13_1.png
                new_img_name = f"{base_name}_{img_file}"
                dst = os.path.join(IMAGES_DIR, new_img_name)
                shutil.move(src, dst)
            
            # Üres ideiglenes mappa törlése
            os.rmdir(files_dir)
            
        # 3. Markdown fájlban lévő hivatkozások okos frissítése
        if os.path.exists(md_file):
            with open(md_file, 'r', encoding='utf-8') as file:
                content = file.read()
            
            # Jupyter ezt generálja: ![png](01_data_preparation_files/output_X_Y.png)
            # Mi erre cseréljük: ![png](images/01_data_preparation_output_X_Y.png)
            pattern = rf'!\[([^\]]*)\]\({base_name}_files/(.*?)\)'
            replacement = rf'![\1](images/{base_name}_\2)'
            new_content = re.sub(pattern, replacement, content)
            
            with open(md_file, 'w', encoding='utf-8') as file:
                file.write(new_content)
                
        print(f"[{notebook}] Kész! [OK]\n")

if __name__ == "__main__":
    print("Dokumentáció frissítése elindult...\n" + "="*40)
    update_documentation()
    print("="*40 + "\nMinden markdown és képhivatkozás sikeresen generálva!")