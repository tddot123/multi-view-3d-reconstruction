"""Verify the formal v1 -> v2 file inventory and MSBuild resource references."""
from pathlib import Path
import hashlib
import json
import re
import subprocess
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
TEXT_SUFFIXES = {'.cpp', '.md', '.json', '.sln', '.vcxproj', '.filters'}


def digest(data, path):
    if Path(path).suffix in TEXT_SUFFIXES or path == '.gitignore':
        data = data.replace(b'\r\n', b'\n')
    return hashlib.sha256(data).hexdigest()


def main():
    manifest = json.loads((ROOT / 'docs/migration-manifest.json').read_text(encoding='utf-8'))
    expected = set(subprocess.check_output(
        ['git', 'ls-tree', '-r', '--name-only', manifest['v1_commit']], cwd=ROOT, text=True
    ).splitlines())
    entries = manifest['files']
    assert {e['v1_path'] for e in entries} == expected, 'v1 inventory mismatch'
    assert len({e['v2_path'] for e in entries}) == len(entries), 'duplicate destination'
    images = 0
    for entry in entries:
        target = ROOT / entry['v2_path']
        assert target.is_file(), f'Missing {target}'
        original = subprocess.check_output(
            ['git', 'show', f"{manifest['v1_commit']}:{entry['v1_path']}"], cwd=ROOT
        )
        assert digest(original, entry['v1_path']) == entry['v1_git_sha256'], entry['v1_path']
        actual = target.read_bytes()
        assert digest(actual, entry['v2_path']) == entry['v2_sha256'], entry['v2_path']
        if target.suffix in {'.png', '.jpg'}:
            assert actual == original, f'Changed sample {target}'
            images += 1

    for old, archived in [('README.md', 'README.v1.md'), ('openGL_enhance/V5_NOTES.md', 'V5_NOTES.v1.md')]:
        original = subprocess.check_output(['git', 'show', f"{manifest['v1_commit']}:{old}"], cwd=ROOT)
        assert digest(original, old) == digest((ROOT / 'docs/archive' / archived).read_bytes(), archived)

    ns = {'m': 'http://schemas.microsoft.com/developer/msbuild/2003'}
    project_path = ROOT / 'build/vs/multi-view-3d-reconstruction.vcxproj'
    project = ET.parse(project_path)
    for tag in ['ClCompile', 'None']:
        for node in project.findall(f'.//m:{tag}[@Include]', ns):
            pattern = node.attrib['Include'].replace('\\', '/')
            assert list(project_path.parent.glob(pattern)), f'Broken project reference: {pattern}'
    solution = (ROOT / 'multi-view-3d-reconstruction.sln').read_text(encoding='utf-8')
    assert 'build\\vs\\multi-view-3d-reconstruction.vcxproj' in solution
    assert '|x86' not in solution
    for p in [ROOT / 'README.md', *ROOT.glob('docs/*.md')]:
        for link in re.findall(r'\]\(([^)]+)\)', p.read_text(encoding='utf-8')):
            if '://' not in link and not link.startswith('#'):
                assert (p.parent / link.split('#')[0]).exists(), f'Broken link: {p}: {link}'
    print(f'PASS: {len(entries)} original files mapped; {images} images byte-identical; original docs preserved; project references and documentation links resolve.')


if __name__ == '__main__':
    main()
