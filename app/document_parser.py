"""Extraction sans réseau, exécutée dans un processus à mémoire/temps bornés."""
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import zipfile
import xml.etree.ElementTree as ET

MAX_TEXT = 1_000_000


def extract(raw, extension):
    if extension == '.pdf':
        from pypdf import PdfReader
        reader = PdfReader(io.BytesIO(raw), strict=True)
        if reader.is_encrypted or len(reader.pages) > 100:
            raise ValueError('PDF chiffré ou supérieur à 100 pages.')
        pages = []
        with tempfile.TemporaryDirectory(prefix='kira_doc_') as folder:
            source = Path(folder) / 'source.pdf'
            source.write_bytes(raw)
            for i, page in enumerate(reader.pages, 1):
                text = (page.extract_text() or '').strip()
                if not text:
                    image = Path(folder) / 'page'
                    subprocess.run(['pdftoppm', '-f', str(i), '-l', str(i), '-singlefile', '-scale-to', '1600', '-png', str(source), str(image)],
                                   check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=15)
                    text = subprocess.check_output(['tesseract', str(image) + '.png', 'stdout', '-l', 'fra+eng'], stderr=subprocess.DEVNULL, timeout=15).decode('utf-8')
                pages.append({'number': i, 'location': 'page', 'content': text[:40000]})
                if sum(len(p['content']) for p in pages) > MAX_TEXT:
                    raise ValueError('Document trop long.')
        return pages
    if extension == '.docx':
        with zipfile.ZipFile(io.BytesIO(raw)) as archive:
            if sum(i.file_size for i in archive.infolist()) > 8_000_000:
                raise ValueError('Document décompressé trop volumineux.')
            root = ET.fromstring(archive.read('word/document.xml'))
            ns = {'w': 'http://schemas.openxmlformats.org/wordprocessingml/2006/main'}
            text = '\n'.join(''.join(n.itertext()) for n in root.findall('.//w:t', ns))
    elif extension in ('.txt', '.md'):
        text = raw.decode('utf-8-sig')
    else:
        raise ValueError('Format non pris en charge.')
    if len(text) > MAX_TEXT:
        raise ValueError('Document trop long.')
    return [{'number': i // 4000 + 1, 'location': 'section', 'content': text[i:i+4000]} for i in range(0, len(text), 4000)]


def main():
    try:
        if os.name == 'posix':
            import resource
            resource.setrlimit(resource.RLIMIT_AS, (512 * 1024**2, 512 * 1024**2))
            resource.setrlimit(resource.RLIMIT_CPU, (55, 55))
            resource.setrlimit(resource.RLIMIT_FSIZE, (20 * 1024**2, 20 * 1024**2))
        raw = sys.stdin.buffer.read(4_000_001)
        if len(raw) > 4_000_000:
            raise ValueError('Document trop volumineux.')
        pages = extract(raw, sys.argv[1])
        if not pages or not any(p['content'].strip() for p in pages):
            raise ValueError('Aucun texte lisible.')
        print(json.dumps({'pages': pages}, ensure_ascii=False))
    except Exception:
        print(json.dumps({'error': 'Extraction impossible : document illisible, protégé, trop long ou OCR indisponible.'}))
        return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
