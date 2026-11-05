"""Masked, session-only Fireworks key entry. No file/registry writes or model calls."""
import argparse
import getpass
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
GREETING = """Write-Host 'Weight Space Lab benchmark terminal' -ForegroundColor Cyan
Write-Host 'Fireworks key loaded only in this terminal and its children. Close it to discard this copy.'
Write-Host 'No key file was created and no model request has started.'
Write-Host ''
Write-Host 'Prepare the comparison:'
Write-Host '  .\\.venv\\Scripts\\python.exe lab.py benchmark create runs/cli-comparison --config experiments/editing-benchmark/cli.example.json'
Write-Host 'Run Claude/Codex CLI editors only (uses account allowance):'
Write-Host '  .\\.venv\\Scripts\\python.exe lab.py benchmark run runs/cli-comparison --allow-cli'
Write-Host 'Fireworks paid calls still require a separate --allow-paid-api opt-in.'
"""


def open_terminal(key):
    if not isinstance(key, str) or not key.strip() or any(c in key for c in '\r\n\x00'):
        raise ValueError("Enter a nonempty key without line breaks")
    if os.name != 'nt':
        raise ValueError("This helper opens a Windows terminal; use a local environment variable on other platforms")
    env = os.environ.copy()
    env['FIREWORKS_API_KEY'] = key.strip()
    try:
        return subprocess.Popen(['powershell.exe', '-NoProfile', '-NoExit', '-Command', GREETING], cwd=ROOT,
                                env=env, creationflags=getattr(subprocess, 'CREATE_NEW_CONSOLE', 0))
    finally:
        env.pop('FIREWORKS_API_KEY', None)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--console', action='store_true', help='Use a masked terminal prompt instead of a dialog')
    if parser.parse_args().console:
        if not sys.stdin.isatty():
            raise ValueError('Use a real terminal so the key cannot fall back to visible input')
        key = getpass.getpass('Paste Fireworks API key (hidden): ')
        try:
            open_terminal(key)
        finally:
            key = None
        return
    import tkinter as tk
    from tkinter import messagebox
    window = tk.Tk()
    window.title('Weight Space Lab - Fireworks key')
    window.geometry('560x205')
    window.resizable(False, False)
    tk.Label(window, text='Paste the Fireworks API key below (hidden).\nIt is passed only to a new terminal; no file is written.\nNo model or API request starts automatically.', justify='left').pack(anchor='w', padx=18, pady=(16, 10))
    entry = tk.Entry(window, show='\u2022', width=65)
    entry.pack(padx=18, fill='x')
    entry.focus_set()
    def submit():
        try:
            open_terminal(entry.get())
        except Exception:
            messagebox.showerror('Weight Space Lab', 'The terminal could not be opened. Check that a key was entered and PowerShell is available.')
            return
        entry.delete(0, 'end')
        window.destroy()
    tk.Button(window, text='Open benchmark terminal', command=submit).pack(anchor='w', padx=18, pady=16)
    window.bind('<Return>', lambda event: submit())
    window.mainloop()


if __name__ == '__main__':
    main()
