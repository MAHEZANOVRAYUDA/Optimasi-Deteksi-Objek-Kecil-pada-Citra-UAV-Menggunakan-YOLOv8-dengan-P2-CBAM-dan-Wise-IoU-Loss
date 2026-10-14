import glob

all_refs = {}
for rf in sorted(glob.glob('Jurnal */*.ris')):
    with open(rf, 'r', encoding='utf-8') as f:
        content = f.read()
    entries = content.split('ER  -')
    for e in entries:
        lines = [l.strip() for l in e.strip().splitlines() if l.strip()]
        if not lines: continue
        title = ''
        authors = []
        year = ''
        journal = ''
        doi = ''
        for l in lines:
            if l.startswith('TI  ') or l.startswith('T1  '):
                title = l[4:].strip().lstrip('- ')
            elif l.startswith('AU  ') or l.startswith('A1  '):
                authors.append(l[4:].strip())
            elif l.startswith('PY  ') or l.startswith('Y1  '):
                year = l[4:].strip()[:4]
            elif l.startswith('JO  ') or l.startswith('JF  ') or l.startswith('JA  '):
                journal = l[4:].strip()
            elif l.startswith('DO  '):
                doi = l[4:].strip()
        if title:
            key = title.lower()[:40]
            if key not in all_refs:
                all_refs[key] = {
                    'title': title,
                    'authors': authors,
                    'year': year,
                    'journal': journal,
                    'doi': doi
                }

print(f"Total unique references across RIS: {len(all_refs)}")
for i, (k, ref) in enumerate(all_refs.items(), 1):
    auth_str = ", ".join(ref['authors'][:3]) if ref['authors'] else 'Unknown'
    if len(ref['authors']) > 3:
        auth_str += " et al."
    doi_str = f"https://doi.org/{ref['doi']}" if ref['doi'] else ""
    print(f"[{i}] {auth_str}, \"{ref['title']}\", {ref['journal']}, {ref['year']}. {doi_str}")
