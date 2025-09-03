#!/usr/bin/env python3
"""
Procura ocorrências de Image.fromarray(..., mode=...) e referências a PIL/Image em todo o diretório de testes.
Rode na raiz do repositório. Saída: lista de arquivos e linhas para revisão.
"""
import os
import re
import sys

ROOT = os.getcwd()
patterns = [
    re.compile(r"Image\.fromarray\([^\)]*mode\s*=\s*['\"]([A-Za-z0-9_]+)['\"]", re.IGNORECASE),
    re.compile(r"from\s+PIL\s+import\s+Image"),
    re.compile(r"import\s+PIL\.Image"),
]

def scan(root):
    results = []
    for dirpath, dirnames, filenames in os.walk(root):
        # opcional: limitar busca à pasta tests/ (recomendado)
        if 'tests' not in dirpath and not dirpath.endswith('tests') and not dirpath.startswith(os.path.join(root, 'tests')):
            # continuar mas só procurar em tests para rapidez
            continue
        for fn in filenames:
            if fn.endswith(('.py',)):
                path = os.path.join(dirpath, fn)
                try:
                    with open(path, 'r', encoding='utf-8') as f:
                        for i, line in enumerate(f, start=1):
                            for pat in patterns:
                                m = pat.search(line)
                                if m:
                                    results.append((path, i, line.rstrip('\n')))
                                    break
                except Exception as e:
                    print(f"Erro lendo {path}: {e}", file=sys.stderr)
    return results

if __name__ == "__main__":
    hits = scan(ROOT)
    if not hits:
        print("Nenhuma ocorrência encontrada nas pastas 'tests/'.")
        sys.exit(0)
    print(f"Encontradas {len(hits)} ocorrências (em tests/):")
    for p, ln, text in hits:
        print(f"{p}:{ln}: {text}")
    print("\nRecomendo revisar cada ocorrência e aplicar uma das correções sugeridas:")
    print("- Para 'L': coerce para uint8 antes de Image.fromarray.")
    print("- Para 'RGB': garantir shape (H,W,3) e dtype uint8.")
    print("\nSe quiser, eu gero patches por arquivo para você aplicar e testar (um arquivo por vez).")