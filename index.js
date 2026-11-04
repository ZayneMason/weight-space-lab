import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';

export const name = 'weight-space-lab';
export const inject = ['skills'];

export function apply(ctx) {
  const base = new URL('./skills/weight-space-lab/', import.meta.url);
  const text = readFileSync(new URL('SKILL.md', base), 'utf8').replaceAll('\r\n', '\n');
  const match = /^---\n([\s\S]*?)\n---\n([\s\S]*)$/.exec(text);
  if (!match) throw new Error('Weight Space Lab skill frontmatter is missing');
  const description = /^description: (.+)$/m.exec(match[1])?.[1];
  if (!description) throw new Error('Weight Space Lab skill description is missing');
  ctx.effect(() => ctx.skills.register({
    name, description, content: match[2].trim(), source: 'bundled',
    resourceBase: { kind: 'directory', path: fileURLToPath(base) },
  }));
}
