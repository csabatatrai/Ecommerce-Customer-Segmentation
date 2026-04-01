import os
import re

# A pont jelzi, hogy abban a mappában keressen, ahonnan futtatod (docs mappa)
docs_dir = '.'

# Végigmegyünk az összes markdown fájlon a jelenlegi mappában
for filename in os.listdir(docs_dir):
    if filename.endswith(".md"):
        filepath = os.path.join(docs_dir, filename)
        
        with open(filepath, 'r', encoding='utf-8') as file:
            content = file.read()
        
        # Kicseréljük az ![png](output_...) részt ![png](images/output_...)-ra
        new_content = re.sub(r'!\[png\]\((output_.*?\.png)\)', r'![png](images/\1)', content)
        
        with open(filepath, 'w', encoding='utf-8') as file:
            file.write(new_content)
            
print("Hivatkozások frissítve!")