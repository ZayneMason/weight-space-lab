"""Self-contained local HTML reports rendered from measured artifacts."""
from html import escape
import json
from pathlib import Path

from .evidence import read_json

STYLE = """*{box-sizing:border-box}body{margin:0;background:#f5f7fa;color:#182433;font:16px/1.55 system-ui,sans-serif}
main{max-width:1100px;margin:auto;padding:40px 24px}h1{font-size:36px;line-height:1.15;letter-spacing:-1px;margin:8px 0 16px}
h2{margin:32px 0 12px}p{max-width:85ch}.eyebrow{font-size:12px;font-weight:750;letter-spacing:2px;color:#385570}
.stats{display:flex;flex-wrap:wrap;gap:16px;margin:24px 0}.stat{flex:1;min-width:150px;padding:18px;background:white;border:1px solid #d9e1ea;border-radius:8px}
.stat strong{display:block;font-size:28px}.muted,small{color:#526477}.visual{background:white;border:1px solid #d9e1ea;border-radius:8px;padding:12px}
svg{display:block;width:100%;height:auto}table{width:100%;border-collapse:collapse;background:white}th,td{text-align:left;padding:10px 12px;border-bottom:1px solid #d9e1ea}
.table-wrap{overflow:auto}.fixed{color:#126343}.regressed{color:#ad3434}pre{white-space:pre-wrap;overflow-wrap:anywhere;padding:18px;background:#14283b;color:#f4f7fb;border-radius:8px;font-size:13px}
code{overflow-wrap:anywhere}a{color:#075f92}li{margin:6px 0}button,input{font:inherit}button{padding:8px 16px;border:1px solid #70869b;border-radius:5px;background:white;color:#182433}
.replay-controls{display:flex;flex-wrap:wrap;align-items:center;gap:12px;margin:12px 0}.replay-controls input{flex:1;min-width:120px}
canvas{display:block;max-width:420px;width:100%;aspect-ratio:1;background:#14283b;image-rendering:pixelated;border-radius:4px}
@media(max-width:540px){main{padding:24px 12px}h1{font-size:29px}.stat strong{font-size:24px}th,td{padding:8px}}
"""


def page(title, body, script=""):
    return ("<!doctype html><html lang=\"en\"><head><meta charset=\"utf-8\">"
            '<meta name="viewport" content="width=device-width,initial-scale=1">'
            '<meta http-equiv="Content-Security-Policy" content="default-src \'none\'; style-src \'unsafe-inline\'; script-src \'unsafe-inline\'; img-src data:;">'
            f"<title>{escape(title)}</title><style>{STYLE}</style></head><body><main>{body}</main>{script}</body></html>")


def stats(items):
    return '<div class="stats">' + ''.join(
        f'<div class="stat"><small>{escape(label)}</small><strong>{escape(str(value))}</strong></div>'
        for label, value in items) + '</div>'


def sign_svg(data, baseline, candidate):
    """Two number lines, labeled predictions and correctness for actual examples."""
    xs = [row["x"][0] for row in data["examples"]]
    low, high = min(xs) - 1, max(xs) + 1
    def scale(x):
        return 160 + (x - low) / (high - low) * 640
    parts = ['<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 880 340" role="img" aria-labelledby="plot-title plot-desc">',
             '<title id="plot-title">Sign classifier: actual predictions before and after a prewritten edit</title>',
             '<desc id="plot-desc">Six fixed examples. The baseline misclassifies all six; the candidate fixes all six. Green means correct; red means incorrect. Labels show predicted class.</desc>',
             '<rect width="880" height="340" fill="#ffffff"/>',
             '<text x="30" y="32" fill="#182433" font-family="system-ui" font-size="18" font-weight="700">Prewritten sign-classifier repair</text>',
             '<text x="30" y="57" fill="#526477" font-family="system-ui" font-size="14">Class 0 = negative · Class 1 = positive · green = correct · red = incorrect</text>']
    for label, report, y in (("Baseline", baseline, 125), ("Candidate", candidate, 230)):
        parts += [f'<text x="30" y="{y+5}" fill="#182433" font-family="system-ui" font-size="16">{label}</text>',
                  f'<line x1="160" y1="{y}" x2="800" y2="{y}" stroke="#b4c1cd"/>',
                  f'<line x1="{scale(0):.2f}" y1="{y-31}" x2="{scale(0):.2f}" y2="{y+31}" stroke="#8c9baa" stroke-dasharray="4 4"/>']
        previous_x = None
        for row, prediction in zip(data["examples"], report["examples"]):
            x = scale(row["x"][0]); color = "#16805a" if prediction["correct"] else "#bd4b4b"
            label_y = y + (60 if previous_x is not None and x - previous_x < 42 else 42)
            parts += [f'<circle data-example="{escape(row["id"], quote=True)}" cx="{x:.2f}" cy="{y}" r="11" fill="{color}"/>',
                      f'<text x="{x:.2f}" y="{y+5}" text-anchor="middle" fill="white" font-family="system-ui" font-size="14">{prediction["prediction"]}</text>',
                      f'<text x="{x:.2f}" y="{label_y}" text-anchor="middle" fill="#526477" font-family="system-ui" font-size="14">{row["x"][0]:g}</text>']
            previous_x = x
    parts += ['<text x="480" y="308" text-anchor="middle" fill="#526477" font-family="system-ui" font-size="14">Input value x (zero is the decision boundary)</text>', '</svg>']
    return '\n'.join(parts)


def demo_report(out, summary):
    out = Path(out)
    before, after = read_json(out / "baseline.json"), read_json(out / "candidate.json")
    data, patch = read_json(out / "heldout.json"), read_json(out / "edit.json")
    receipt = read_json(out / "candidate/receipt.json")
    visual = sign_svg(data, before, after)
    (out / "behavior.svg").write_text(visual, encoding="utf-8")
    rows = []
    for example, a, b in zip(data["examples"], before["examples"], after["examples"]):
        outcome = "fixed" if not a["correct"] and b["correct"] else "regressed" if a["correct"] and not b["correct"] else "unchanged"
        rows.append(f'<tr><td>{escape(example["id"])}</td><td>{example["x"][0]:g}</td><td>{a["label"]}</td><td>{a["prediction"]}</td><td>{b["prediction"]}</td><td class="{outcome}">{outcome}</td></tr>')
    links = ["original.npz", "architecture.json", "heldout.json", "edit.json", "candidate/patch.json",
             "candidate/receipt.json", "baseline.json", "candidate.json", "comparison.json", "experiment.json"]
    body = '<p class="eyebrow">WEIGHT SPACE LAB · OFFLINE CPU DEMONSTRATION</p><h1>A two-weight repair, with evidence.</h1>'
    body += '<p>This deterministic example uses a <strong>prewritten patch</strong> to reverse an inverted sign classifier. It demonstrates inspection, editing, evaluation and evidence review; it is not agent discovery or a benchmark.</p>'
    body += stats([("Baseline correct", f'{sum(r["correct"] for r in before["examples"])}/{before["samples"]}'),
                   ("Candidate correct", f'{sum(r["correct"] for r in after["examples"])}/{after["samples"]}'),
                   ("Stored entries changed", f'{receipt["changed"]}/{receipt["total"]}'),
                   ("Fixed / regressed", f'{len(summary["fixed_ids"])} / {len(summary["regressed_ids"])}')])
    body += f'<div class="visual">{visual}</div><p class="muted">The six predefined examples are illustrative. The legacy filename <code>heldout.json</code> does not make these an independent unseen test set.</p>'
    body += '<h2>Observed behavior</h2><div class="table-wrap"><table><thead><tr><th>Example</th><th>Input</th><th>Expected</th><th>Baseline</th><th>Candidate</th><th>Change</th></tr></thead><tbody>' + ''.join(rows) + '</tbody></table></div>'
    body += f'<h2>Exact proposed patch</h2><p>{escape(receipt["reason"])}</p><pre>{escape(json.dumps(patch, indent=2))}</pre>'
    body += f'<h2>Checkpoint identity</h2><p>Source SHA-256<br><code>{receipt["base_sha256"]}</code></p><p>Candidate SHA-256<br><code>{receipt["candidate_sha256"]}</code></p><p>The source remains unchanged. Selecting the original checkpoint is rollback.</p>'
    body += '<h2>Review the records</h2><ul>' + ''.join(f'<li><a href="{name}">{name}</a></li>' for name in links) + '</ul><p>Run <code>python lab.py verify OUTPUT_DIRECTORY</code> from the toolkit checkout to verify hashes and recompute the demo evidence. The ledger is unsigned.</p>'
    (out / "report.html").write_text(page("Weight Space Lab — offline sign demo", body), encoding="utf-8")


def breakout_report(out, report, replay, scope):
    out = Path(out)
    scope_label = "SMOKE CHECK — NOT A FULL BENCHMARK" if scope == "smoke" else "FULL 1,000-EPISODE PROTOCOL"
    body = f'<p class="eyebrow">WEIGHT SPACE LAB · {scope_label}</p><h1>MinAtar Breakout</h1>'
    body += '<p>A local evaluation of the published network from Takehi et al. No fresh model generation or hosted model allowance was used. Scores describe this recorded protocol and random seeds.</p>'
    body += stats([("Episodes", report["episodes"]), ("Mean score", f'{report["mean"]:.3f}'),
                   ("Standard error", f'{report["se"]:.3f}'), ("Step ceiling", report["settings"]["max_steps"])])
    body += f'<h2>Episode 0 replay</h2><p>First {len(replay["frames"])} steps of episode 0, selected in advance. That evaluated episode scored {replay["episode"]["score"]} over {replay["episode"]["steps"]} steps. The preview may end before the episode.</p>'
    body += '<canvas id="game" width="400" height="400" aria-label="MinAtar Breakout recorded gameplay preview"></canvas><div class="replay-controls"><button id="play" type="button">Play</button><label for="frame">Frame</label><input id="frame" type="range" min="0" value="0"><output id="frame-label" aria-live="polite"></output></div><p class="muted">Teal: paddle · Orange: ball · Gray: previous ball position · Blue: bricks</p>'
    body += f'<h2>Inputs and protocol</h2><p>Checkpoint SHA-256<br><code>{report["checkpoint_sha256"]}</code></p><pre>{escape(json.dumps(report["settings"], indent=2))}</pre>'
    body += '<p>Original research evaluation seeds/backend were not released. This is a statistical reproduction attempt. Smoke results must not be compared to the published 1,000-episode score as though the protocols match.</p>'
    body += '<h2>Evidence</h2><ul>' + ''.join(f'<li><a href="{name}">{name}</a></li>' for name in ["evaluation.json", "comparison.json", "verification.json", "replay.json", "reference/sources.json", "reference/LICENSE", "reference/LICENSE-MinAtar", "experiment.json"]) + '</ul>'
    body += '<p>Research attribution: Takehi et al., Experiment 5, MinAtar Breakout. Source revision <code>' + escape(replay["research_revision"]) + '</code>. Upstream licenses are retained under <code>reference/</code>.</p>'
    frames = json.dumps(replay["frames"], separators=(",", ":")).replace("<", "\\u003c")
    script = '''<script>
const frames=FRAMES, canvas=document.getElementById('game'),ctx=canvas.getContext('2d');
const slider=document.getElementById('frame'),label=document.getElementById('frame-label'),play=document.getElementById('play');
slider.max=frames.length-1;let timer=null;
function draw(){const f=frames[Number(slider.value)];ctx.fillStyle='#14283b';ctx.fillRect(0,0,400,400);
for(let i=0;i<100;i++){const bits=parseInt(f.cells[i],16);let color=null;if(bits&8)color='#5d9bdf';if(bits&4)color='#70869b';if(bits&1)color='#20c7ac';if(bits&2)color='#ffb454';if(color){ctx.fillStyle=color;ctx.fillRect(i%10*40+1,Math.floor(i/10)*40+1,38,38);}}
label.textContent=`${Number(slider.value)+1}/${frames.length} · step ${f.step} · score ${f.score}`;}
function stop(){clearInterval(timer);timer=null;play.textContent='Play';}
slider.addEventListener('input',()=>{stop();draw();});play.addEventListener('click',()=>{if(timer){stop();return;}if(Number(slider.value)===frames.length-1)slider.value=0;play.textContent='Pause';timer=setInterval(()=>{if(Number(slider.value)>=frames.length-1){stop();return;}slider.value=Number(slider.value)+1;draw();},80);});draw();
</script>'''.replace("FRAMES", frames)
    (out / "report.html").write_text(page("Weight Space Lab — MinAtar Breakout", body, script), encoding="utf-8")


def controller_report(out, result, state):
    comparison = result["comparison"]
    actor = result["actor"]
    body = '<p class="eyebrow">WEIGHT SPACE LAB · INDEPENDENT CONTROLLED EXPERIMENT</p><h1>Bounded threshold repair</h1>'
    body += f'<p>Editor type: <strong>{escape(actor["actor_type"])}</strong>. Identifier: {escape(actor["editor_id"])} (caller-declared). Manual verification is not agent discovery. This study is separate from upstream reproductions.</p>'
    body += stats([("Baseline accuracy", f'{comparison["baseline_accuracy"]:.1%}'),
                   ("Frozen candidate accuracy", f'{comparison["candidate_accuracy"]:.1%}'),
                   ("Fixed / regressed", f'{len(comparison["fixed_ids"])} / {len(comparison["regressed_ids"])}'),
                   ("Predeclared criterion", "MET" if result["success_rule_met"] else "NOT MET")])
    body += '<h2>Frozen candidate and matched final test</h2><p>Both models used the same held-out examples, architecture and evaluator settings. The candidate was copied and hashed before final evaluation. No editing or practice feedback is permitted afterward.</p>'
    body += f'<p>Source SHA-256<br><code>{comparison["baseline_sha256"]}</code></p><p>Frozen candidate SHA-256<br><code>{comparison["candidate_sha256"]}</code></p>'
    body += '<h2>Budgets and complete outcomes</h2><pre>' + escape(json.dumps(result, indent=2)) + '</pre>'
    body += '<h2>Every practice-feedback outcome</h2><div class="table-wrap"><table><thead><tr><th>Candidate</th><th>Baseline</th><th>Candidate</th><th>Fixed</th><th>Regressed</th></tr></thead><tbody>'
    for event in state["events"]:
        if event["action"] == "feedback" and event["outcome"] == "complete":
            feedback = read_json(out.parent / event["record"])
            body += (f'<tr><td>{escape(event["candidate_id"])}</td><td>{feedback["baseline_accuracy"]:.1%}</td>'
                     f'<td>{feedback["candidate_accuracy"]:.1%}</td><td>{len(feedback["fixed_ids"])}</td>'
                     f'<td>{len(feedback["regressed_ids"])}</td></tr>')
    body += '</tbody></table></div>'
    failures = [e for e in state["events"] if e["outcome"] in ("failed", "rejected", "denied")]
    body += '<h2>Rejected, failed, and denied actions</h2><pre>' + escape(json.dumps(failures, indent=2)) + '</pre>'
    body += '<h2>Event record</h2><div class="table-wrap"><table><thead><tr><th>Event</th><th>Action</th><th>Outcome</th><th>Candidate</th></tr></thead><tbody>'
    body += ''.join(f'<tr><td>{e["index"]}</td><td>{escape(e["action"])}</td><td>{escape(e["outcome"])}</td><td>{escape(e.get("candidate_id", ""))}</td></tr>' for e in state["events"])
    body += '</tbody></table></div><p>The authoritative state.json adds final completion and its result hashes after this report is written. Hash ledgers are unsigned; OS isolation and editor provenance require independent verification.</p>'
    body += '<h2>Records</h2><ul>' + ''.join(f'<li><a href="{path}">{path}</a></li>' for path in ("baseline.json", "candidate.json", "result.json", "../protocol.json", "../state.json", "../frozen/selection.json")) + '</ul>'
    Path(out / "report.html").write_text(page("Weight Space Lab — bounded threshold repair", body), encoding="utf-8")
