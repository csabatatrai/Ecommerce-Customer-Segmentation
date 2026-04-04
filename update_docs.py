"""
Jupyter notebookok → Markdown dokumentacio, leiro kepnevekkel.

Normal hasznalat (minden notebook vegen automatikusan fut):
    python update_docs.py --notebook 03_churn_prediction.ipynb

Teljes dokimentacio ujraepites (ha abraat toroltel, vagy a script logikajat modositottad):
    python update_docs.py --clean

A --clean flag torli a teljes docs/ mappat, majd nullarol ujrageneralja.
Normal fejlesztes kozben nem szukseges — csak akkor, ha elavult fajlokat
kell eltavolitani a docs/-bol.
"""

import os
import re
import json
import shutil
import argparse
import subprocess


# ---------------------------------------------------------------------------
# Konfiguráció
# ---------------------------------------------------------------------------
NOTEBOOKS_DIR = '.'
DOCS_DIR      = 'docs'
IMAGES_BASE   = os.path.join(DOCS_DIR, 'images')

# Opcionális kézi elnevezések.
# Kulcs:  "{notebook_alap_neve}/{jupyter_stem}"
#         pl. "03_churn_prediction/output_37_1"
#         vagy "03_churn_prediction/03_churn_prediction_37_1"  (ujabb nbconvert)
# Ertek:  leiro nev kiterjesztes nelkul, a kepfile mappajan belul
#         pl. "03_8.3._Precision_recall_gorbe"
#
# Ha egy kephez nincs bejegyzes, az automatikus section-cim alapu nev lep eletbe.
# Ha az sem talalhato, figyelmeztes + fallback nev keletkezik.
#
IMAGE_NAME_MAP: dict[str, str] = {
    # "03_churn_prediction/output_37_1": "03_8.3._Precision_recall_gorbe",
    # "03_churn_prediction/output_29_1": "03_9.4._SHAP_summary_globalis",
}


# ---------------------------------------------------------------------------
# Névépítés
# ---------------------------------------------------------------------------

def sanitize_title(title: str) -> str:
    """
    Fajlnevbe illesztheto cimet keszit: szokozok → alsovonas,
    Windows-invalid karakterek (':?*<>"|\\/' es em-dash) eltavolitva,
    ekezetes betuk megmaradnak.
    """
    # Vezeto gondolatjel/kotojel eltavolitasa (pl. "– CV eredmenyek" → "CV eredmenyek")
    title = re.sub(r'^[\s\-\u2013\u2014]+', '', title).strip()
    # Windows-on tiltott karakterek torlese
    title = re.sub(r'[\\/:*?"<>|]', '', title)
    # Zarqjelek es vesszok megtartasa, de szokoz → alsovonas
    title = title.strip().replace(' ', '_')
    return title


def format_image_name(nb_number: str, section_num: str, section_title: str, count: int) -> str:
    """
    Összerakja a képfájl nevét a notebook sorszámából, a szekció számból
    és a szekció címéből. Szóközök helyett alsóvonás, ékezetek megmaradnak.

    Példák:
      nb_number="01", section_num="1.5.", title="Cutoff validáció", count=0
        → "01_1.5._Cutoff_validáció"

      nb_number="03", section_num="9.4.", title="SHAP Waterfall plot", count=1
        → "03_9.4._SHAP_Waterfall_plot_2"  (második kép ugyanabból a szekcióból)

      nb_number="02", section_num="", title="Bevezetes", count=0
        → "02_Bevezetes"  (szám nélküli fejléc esetén)
    """
    title_part = sanitize_title(section_title)
    if section_num:
        base = f"{nb_number}_{section_num}_{title_part}"
    else:
        base = f"{nb_number}_{title_part}"
    suffix = f"_{count + 1}" if count > 0 else ""
    return f"{base}{suffix}"


def build_auto_name_map(notebook_path: str) -> dict[str, str]:
    """
    Notebookot elemez és visszaad egy  { "output_X_Y": "leiro_nev" }  szótárt.

    A Jupyter nbconvert képfájlokat az alábbi névformátumok egyikével generálja:
      - régi nbconvert:  output_{cell_idx}_{out_idx}.png
      - új nbconvert:    {notebook_neve}_{cell_idx}_{out_idx}.png
    Mindkét formátumhoz bejegyezzük a leíró nevet.

    cell_idx  = a cella 0-alapú sorszáma az ÖSSZES cellán belül (md + code)
    out_idx   = az output 0-alapú indexe az adott cellán belüli outputs listán

    A nevet a megelőző Markdown-fejléc adja, a következő formában:
      {notebook_szam}_{szekcioszam}_{szekciocim_alsovonas}
    pl. "01_1.5._Cutoff_validáció"
    """
    nb_stem   = os.path.basename(notebook_path).removesuffix('.ipynb')
    nb_number = nb_stem.split('_')[0]   # "01" a "01_data_preparation"-ból

    with open(notebook_path, encoding='utf-8') as f:
        nb = json.load(f)

    auto_map: dict[str, str] = {}
    current_section_num   = ''
    current_section_title = 'output'
    section_counter: dict[str, int] = {}

    for cell_idx, cell in enumerate(nb.get('cells', [])):
        cell_type = cell.get('cell_type', '')

        # Markdown cella → frissítjük az aktív szekció-azonosítót
        if cell_type == 'markdown':
            source = ''.join(cell.get('source', []))
            for line in source.splitlines():
                stripped = line.strip()
                if stripped.startswith('#'):
                    heading = re.sub(r'^#+\s*', '', stripped).strip()
                    # Szekciószám kinyerése: "1.5. Cutoff validáció" → ("1.5.", "Cutoff validáció")
                    m = re.match(r'^([\d]+(?:\.[\d]*)*\.?)\s+(.*)', heading)
                    if m:
                        current_section_num   = m.group(1)
                        current_section_title = m.group(2).strip()
                    else:
                        current_section_num   = ''
                        current_section_title = heading
                    break

        # Code cella → képes outputokat indexeljük
        elif cell_type == 'code':
            for out_idx, output in enumerate(cell.get('outputs', [])):
                data = output.get('data', {})
                is_image = any(
                    mime in data
                    for mime in ('image/png', 'image/jpeg', 'image/svg+xml', 'image/gif')
                )
                if not is_image:
                    continue

                section_key = f"{current_section_num}|{current_section_title}"
                count = section_counter.get(section_key, 0)
                section_counter[section_key] = count + 1

                name = format_image_name(
                    nb_number, current_section_num, current_section_title, count
                )

                # Mindkét nbconvert névformátumhoz bejegyezzük
                auto_map[f"output_{cell_idx}_{out_idx}"]    = name
                auto_map[f"{nb_stem}_{cell_idx}_{out_idx}"] = name

    return auto_map


def resolve_image_name(
    base_name: str,
    jupyter_stem: str,
    auto_map: dict[str, str],
) -> tuple[str, bool]:
    """
    Leiro fajlnevet ad vissza (kiterjesztes nelkul) es egy bool-t (True = fallback).

    Prioritas:
      1. IMAGE_NAME_MAP["{base_name}/{jupyter_stem}"]   <- kezi override
      2. auto_map[jupyter_stem]                          <- automatikus szekciocim
      3. fallback: {nb_szam}_{jupyter_stem}              <- figyelmeztessel
    """
    manual_key = f"{base_name}/{jupyter_stem}"

    if manual_key in IMAGE_NAME_MAP:
        return IMAGE_NAME_MAP[manual_key], False

    if jupyter_stem in auto_map:
        return auto_map[jupyter_stem], False

    # Fallback: notebook szam + eredeti Jupyter stem
    nb_number = base_name.split('_')[0]
    return f"{nb_number}_{jupyter_stem}", True


def safe_dst(path: str) -> str:
    """Ha a célfájl már létezik, _2, _3, ... szuffix hozzáadásával egyedi nevet keres."""
    if not os.path.exists(path):
        return path
    stem, ext = os.path.splitext(path)
    counter = 2
    while os.path.exists(f"{stem}_{counter}{ext}"):
        counter += 1
    return f"{stem}_{counter}{ext}"


# ---------------------------------------------------------------------------
# Fő logika
# ---------------------------------------------------------------------------

def update_documentation(clean: bool = False, only: str | None = None) -> None:
    """
    .ipynb fajlokat Markdown-na konvertal, a kepeket
    docs/images/{notebook_neve}/ almappaba helyezi leiro nevvel,
    es frissiti a .md hivatkozasokat.

    only: ha meg van adva, csak ezt az egy notebookot dolgozza fel
          (pl. "03_churn_prediction.ipynb") — a tobbi erintetlen marad.
    """
    if clean and os.path.exists(DOCS_DIR):
        shutil.rmtree(DOCS_DIR)
        print(f"[clean] '{DOCS_DIR}/' torolve.\n")

    os.makedirs(IMAGES_BASE, exist_ok=True)

    if only:
        if not os.path.exists(os.path.join(NOTEBOOKS_DIR, only)):
            print(f"Nem talaltam: '{only}'")
            return
        notebooks = [only]
    else:
        notebooks = sorted(f for f in os.listdir(NOTEBOOKS_DIR) if f.endswith('.ipynb'))

    if not notebooks:
        print("Nem talaltam .ipynb fajlt.")
        return

    for notebook in notebooks:
        base_name  = notebook.removesuffix('.ipynb')
        md_file    = os.path.join(DOCS_DIR, f"{base_name}.md")
        files_dir  = os.path.join(DOCS_DIR, f"{base_name}_files")  # nbconvert temp
        nb_img_dir = os.path.join(IMAGES_BASE, base_name)          # vegleges almappa

        print(f"[{notebook}] Konvertalas Markdown-ra...")

        # 1. nbconvert futtatasa (sajat outputja elnyomva, script maga logol)
        subprocess.run(
            ["jupyter", "nbconvert", "--to", "markdown", notebook, "--output-dir", DOCS_DIR],
            check=True,
            capture_output=True,
        )

        # 2. Automatikus nevsablon a notebook szekciociimeibol
        auto_map = build_auto_name_map(notebook)

        # 3. Kepek athelyezese es atnevezese
        # Elobb toroljuk a korabbi kepeket — kulonben ujrafutaskor duplikalodnak
        if os.path.exists(nb_img_dir):
            shutil.rmtree(nb_img_dir)
        os.makedirs(nb_img_dir)
        ref_map: dict[str, str] = {}  # regi md-hivatkozas → uj md-hivatkozas
        fallback_count = 0

        if os.path.exists(files_dir):
            for img_file in sorted(os.listdir(files_dir)):
                stem, ext = os.path.splitext(img_file)
                src = os.path.join(files_dir, img_file)

                descriptive, is_fallback = resolve_image_name(base_name, stem, auto_map)

                # Move elobb — a print crash nem akadalyozhatja meg az athelyezest
                new_filename = f"{descriptive}{ext}"
                dst = os.path.join(nb_img_dir, new_filename)
                shutil.move(src, dst)

                if is_fallback:
                    fallback_count += 1
                    print(
                        f"  [!] Ismeretlen kep: '{base_name}/{stem}'\n"
                        f"      Fallback nevvel mentve: {new_filename}\n"
                        f"      Ha leiro nevet szeretnel, add hozza:\n"
                        f"      IMAGE_NAME_MAP[\"{base_name}/{stem}\"] = \"<nev>\","
                    )

                # Hivatkozas-par a .md cserhez
                # Jupyter generalja:  ![png](03_churn_prediction_files/output_X_Y.png)
                # Mi erre csereljuk:  ![png](images/03_churn_prediction/03_9.4._SHAP....png)
                old_ref = f"{base_name}_files/{img_file}"
                new_ref = f"images/{base_name}/{os.path.basename(dst)}"
                ref_map[old_ref] = new_ref

            # Ideiglenes mappa torlese — rmtree, mert az nbconvert
            # nem-kep fajlokat is generalhat, az rmdir csak ures mappakon mukodik
            shutil.rmtree(files_dir, ignore_errors=True)

        # 4. Hivatkozasok frissitese a .md fajlban (string-replace, nem regex)
        if os.path.exists(md_file) and ref_map:
            with open(md_file, encoding='utf-8') as f:
                content = f.read()
            for old_ref, new_ref in ref_map.items():
                content = content.replace(old_ref, new_ref)
            with open(md_file, 'w', encoding='utf-8') as f:
                f.write(content)

        status = f"{len(ref_map)} kep"
        if fallback_count:
            status += f", {fallback_count} db fallback [!]"
        print(f"[{notebook}] [OK] Kesz! ({status})\n")


# ---------------------------------------------------------------------------
# Belépési pont
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Jupyter notebookok → Markdown docs, leiro kepnevekkel"
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Torolje a docs/ mappat teljesen ujrageneralas elott",
    )
    parser.add_argument(
        "--notebook",
        metavar="FAJL",
        help="Csak ezt az egy notebookot dolgozza fel, pl. --notebook 03_churn_prediction.ipynb",
    )
    args = parser.parse_args()

    print("Docs frissitese...\n" + "=" * 50)
    update_documentation(clean=args.clean, only=args.notebook)
    print("=" * 50 + "\nKesz!")