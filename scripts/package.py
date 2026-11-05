"""Build standalone Codex ZIP and DSH tarball without npm lifecycle scripts."""
import hashlib
import json
from pathlib import Path
import tarfile
import zipfile

root = Path(__file__).resolve().parents[1]
manifest = json.loads((root / "package.json").read_text())
files = [root / name for name in ("package.json", "README.md", "README.zh.md", "RESULTS.md", "LICENSE", "NOTICE.md", "lab.py", "requirements.txt", "requirements-breakout.txt")]
for name in (".codex-plugin", "skills", "locale", "assets", "qwen", "docs", "evidence", "licenses", "weight_space_lab", "experiments"):
    files += [p for p in (root / name).rglob("*") if p.is_file() and "__pycache__" not in p.parts and p.suffix != ".pyc"]
files += [root / "index.js", root / "cordis.patch.yml", root / "scripts/check_controller.py", root / "scripts/check_editor_session.py", root / "scripts/sanitize_qwen_pilot.py"]
files = sorted(set(files))
output = root / "dist"
output.mkdir(exist_ok=True)
zip_path = output / f"weight-space-lab-{manifest['version']}-codex.zip"
tar_path = output / f"dsh-weight-space-lab-{manifest['version']}.tgz"
with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as archive:
    for path in files:
        archive.write(path, path.relative_to(root))
with tarfile.open(tar_path, "w:gz") as archive:
    for path in files:
        archive.add(path, arcname=f"package/{path.relative_to(root)}", recursive=False)
checksums = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in (zip_path, tar_path)}
(output / "checksums.json").write_text(json.dumps(checksums, indent=2) + "\n")
print(json.dumps(checksums, indent=2))
