// Run under the host's Node/Electron runtime with its node_modules path and
// the installed bundle path as arguments. No profile or model credentials read.
const assert = require('node:assert/strict');
const { join } = require('node:path');
const { pathToFileURL } = require('node:url');
const { existsSync } = require('node:fs');

(async () => {
  const [runtime, installed] = process.argv.slice(2);
  if (!runtime || !installed) throw new Error('Usage: runtime-smoke.cjs RUNTIME_NODE_MODULES INSTALLED_BUNDLE');
  const { Context } = require(join(runtime, '@deepseek-ai/cordis'));
  const { SkillRegistry, renderSkillContent } = require(join(runtime, '@deepseek-ai/dsh-skill'));
  const plugin = await import(pathToFileURL(join(installed, 'index.js')));
  const ctx = new Context();
  const registry = ctx.plugin(SkillRegistry);
  await registry;
  const fiber = ctx.plugin(plugin);
  await fiber;
  const skills = ctx.get('skills');
  const skill = await skills.get('weight-space-lab');
  assert(skill, 'Packaged skill must load in the actual host registry');
  assert.equal(skill.invocation.modelInvocable, true);
  assert.equal(skill.invocation.userInvocable, true);
  assert(existsSync(join(skill.resourceBase.path, 'scripts/weight_lab.py')));
  const rendered = renderSkillContent(skill);
  assert(rendered.includes('<skill_content name="weight-space-lab">'));
  assert(rendered.includes('Boundaries'));
  assert(rendered.includes('Transformer models require'));
  console.log(JSON.stringify({loaded: skill.name, provider: skill.provider, modelInvocable: true, userInvocable: true, rendered: true}));
})().catch(error => { console.error(error); process.exitCode = 1; });
