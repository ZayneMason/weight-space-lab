# Plugin installation

The root `lab.py` entry point and the existing helper CLIs work without installing
the plugin into an active account/profile. The plugin adds shared instructions
for Codex and DeepSeek Harness; it does not change a host assistant's weights.

## Codex

The package contains `.codex-plugin/plugin.json` and shared `skills/` resources.
Add the package to a local plugin marketplace using the
[official packaging instructions](https://developers.openai.com/plugins/build/plugins),
then install it through the Plugins UI. An unpacked skill directory can also be
used with skill discovery. The shared entry point is at the package root; the
original `skills/weight-space-lab/scripts/` commands remain available.

## DeepSeek Harness

Use `plugin_manager` with its `install_bundle` action and the repository directory,
or the CLI after choosing the actual checkout path:

```bash
dsh plugin --profile desktop add /absolute/path/to/weight-space-lab
```

Read the install result and restart when requested. The plugin registers a skill
through the `skills` service; no preset is overridden. A shell-enabled host preset
is required to run the Python tools. Active-profile installation is not part of
the local verification for this change.

## Packages

```bash
python scripts/package.py
```

The builder writes Codex ZIP and DSH tarball packages under `dist/`, including the
root entry point, shared modules, task adapters, documentation, and licenses.
Generated runs, caches, credentials and model weights are excluded.
