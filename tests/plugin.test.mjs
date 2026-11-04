import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync, existsSync } from 'node:fs';
import { resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { apply, inject } from '../index.js';

test('DSH registers a loadable skill and disposes it on unload', () => {
  const registry = new Map();
  const cleanup = [];
  const ctx = {
    effect: (fn) => cleanup.push(fn()),
    skills: { register(skill) {
      assert(!registry.has(skill.name));
      registry.set(skill.name, skill);
      return () => registry.delete(skill.name);
    } },
  };
  assert.deepEqual(inject, ['skills']);
  apply(ctx);
  const skill = registry.get('weight-space-lab');
  assert.equal(skill.source, 'bundled');
  assert(skill.content.includes('evaluate-mlp'));
  assert(skill.content.includes('Transformer models require'));
  assert(!skill.content.startsWith('---'));
  assert(existsSync(resolve(skill.resourceBase.path, 'scripts/weight_lab.py')));
  assert(existsSync(resolve(skill.resourceBase.path, 'references/cli.md')));
  cleanup.forEach(fn => fn());
  assert.equal(registry.size, 0);
  apply(ctx);
  assert.equal(registry.size, 1);
});

test('Codex and DSH resolve the identical skill and CLI', () => {
  const root = fileURLToPath(new URL('../', import.meta.url));
  const manifest = JSON.parse(readFileSync(resolve(root, '.codex-plugin/plugin.json')));
  assert.equal(manifest.name, 'weight-space-lab');
  assert.equal(manifest.requires_local_executor, true);
  let skill;
  apply({effect: fn => fn(), skills: {register: value => {skill = value; return () => {};}}});
  assert.equal(resolve(root, manifest.skills, 'weight-space-lab'), resolve(skill.resourceBase.path));
  assert(!JSON.stringify(manifest).includes('/Users/'));
});
