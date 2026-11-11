"""Self-contained, expandable HTML explorer for the V07 hierarchy."""
import html
import json


_HTML = r'''<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Trace Mining — Variant 07</title><style>
:root{--bg:#fafaf9;--fg:#1c1917;--mut:#78716c;--card:#fff;--bar:#2563eb;--line:#e7e5e4;--good:#15803d;--warn:#b45309}
@media(prefers-color-scheme:dark){:root{--bg:#1c1917;--fg:#f5f5f4;--mut:#a8a29e;--card:#292524;--bar:#60a5fa;--line:#44403c;--good:#4ade80;--warn:#fbbf24}}
*{box-sizing:border-box}body{background:var(--bg);color:var(--fg);font:14px/1.5 system-ui,sans-serif;margin:0 auto;padding:18px;max-width:1200px}
h1{font-size:22px;margin:2px 0 4px}h2{font-size:17px;margin:22px 0 8px}.mut{color:var(--mut)}
.kpis{display:grid;grid-template-columns:repeat(auto-fit,minmax(145px,1fr));gap:8px;margin:14px 0}
.k,.panel,details{background:var(--card);border:1px solid var(--line);border-radius:9px}.k{padding:10px 12px}.k b{display:block;font-size:18px;line-height:1.35}.k span{font-size:12px;color:var(--mut)}
.panel{padding:12px 14px;margin:10px 0}.toolbar{display:flex;gap:8px;flex-wrap:wrap;align-items:center;margin:12px 0}.toolbar input{flex:1;min-width:220px;padding:9px 11px;border:1px solid var(--line);border-radius:7px;background:var(--card);color:var(--fg)}button{padding:8px 10px;border:1px solid var(--line);border-radius:7px;background:var(--card);color:var(--fg);cursor:pointer}
details{margin:7px 0;padding:7px 10px}details details{margin-left:10px}summary{cursor:pointer;list-style:none;display:grid;grid-template-columns:minmax(180px,1fr) minmax(100px,260px) 95px;gap:9px;align-items:center}summary::-webkit-details-marker{display:none}.bar{height:8px;background:var(--line);border-radius:5px;overflow:hidden}.bar i{display:block;height:100%;background:var(--bar)}
.meta{font-size:12px;color:var(--mut);margin:5px 0 9px}.metrics{display:grid;grid-template-columns:repeat(auto-fit,minmax(125px,1fr));gap:6px;margin:7px 0}.metric{border:1px solid var(--line);border-radius:6px;padding:6px 8px;font-size:12px}.metric b{display:block;font-size:14px}.ex{font-size:12px;margin:6px 0;padding:6px 9px;border-left:3px solid var(--bar);overflow-wrap:anywhere}.ids{font:11px ui-monospace,monospace;overflow-wrap:anywhere;color:var(--mut)}.tag{display:inline-block;border:1px solid var(--line);border-radius:999px;padding:1px 7px;font-size:11px;margin-right:4px}.good{color:var(--good)}.warn{color:var(--warn)}
@media(max-width:640px){body{padding:12px}summary{grid-template-columns:1fr 68px}.bar{display:none}.meta{font-size:11px}}
</style></head><body>
<h1>Trace Mining — Variant 07</h1><div class="mut">Business use case → workflow → trace. Recorded, estimated, and unknown token usage are reported separately.</div>
<div id="kpis" class="kpis"></div><section id="analysis" class="panel"></section>
<div class="toolbar"><input id="search" type="search" placeholder="Filter use cases or workflows"><button id="expand">Expand all</button><button id="collapse">Collapse all</button></div>
<div id="tree"></div><h2>Evidence notes</h2><div id="notes"></div>
<script>
const D=__DATA__;
const esc=v=>String(v??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const f=n=>n==null?'—':Number(n).toLocaleString();
const pct=n=>n==null?'—':`${Number(n).toFixed(2)}%`;
const S=D.summary;
const cards=[['Sessions',S.sessions],['Calls',S.calls],['Direct GLM labels',S.glm_direct],['Local suggestions',S.local_suggestions],['Recorded input',S.recorded_input_tokens],['Recorded output',S.recorded_output_tokens],['Estimated tokens',S.estimated_total_tokens],['Unknown-usage calls',S.unknown_calls]];
document.getElementById('kpis').innerHTML=cards.map(([a,b])=>`<div class="k"><span>${esc(a)}</span><b>${f(b)}</b></div>`).join('');
const V=D.validation||{};const bits=[];
if(V.validation_count!=null)bits.push(`<span class="tag">GLM holdout n=${f(V.validation_count)}</span><span class="tag">Exact use-case matches ${f(V.use_case_correct)}/${f(V.validation_count)}</span><span class="tag">Exact workflow matches ${f(V.workflow_correct)}/${f(V.validation_count)}</span><span class="tag">Use-case macro-F1 ${Number(V.use_case_macro_f1||0).toFixed(3)}</span><span class="tag">Workflow macro-F1 ${Number(V.workflow_macro_f1||0).toFixed(3)}</span>`);
bits.push(`<span class="tag ${D.checks_passed?'good':'warn'}">Structure checks ${D.checks_passed?'passed':'not passed'} (${f(D.check_count)})</span>`);
document.getElementById('analysis').innerHTML=`<b>Run summary</b><div class="meta">${esc(D.input_name)} · ${f(S.archetypes)} archetypes (${f(S.selected_archetypes)} selected) · ${f(S.use_case_count)} use cases</div><div>${bits.join(' ')}</div><div class="meta">${f(S.glm_direct)} sessions have direct GLM labels; ${f(S.local_suggestions)} local model suggestions remain marked <code>needs_review</code>.</div>`;
const row=m=>`<div class="metrics"><div class="metric">Calls<b>${f(m.calls)}</b></div><div class="metric">Sessions<b>${f(m.sessions)}</b></div><div class="metric">Recorded in / out<b>${f(m.recorded_input_tokens)} / ${f(m.recorded_output_tokens)}</b></div><div class="metric">Recorded total<b>${f(m.recorded_total_tokens)}</b></div><div class="metric">Estimated total<b>${f(m.estimated_total_tokens)}</b></div><div class="metric">Unknown calls<b>${f(m.unknown_calls)}</b></div><div class="metric">Recorded-token share<b>${pct(m.recorded_share)}</b></div></div>`;
const summary=(name,share,count)=>`<summary><span>${name}</span><span class="bar"><i style="width:${Math.max(0,Math.min(100,Number(share)||0))}%"></i></span><span>${pct(share)} <span class="mut">(${f(count)})</span></span></summary>`;
document.getElementById('tree').innerHTML=D.business_use_cases.map(u=>`<details class="node"><summary>${esc(u.name)}<span class="bar"><i style="width:${Math.max(0,Math.min(100,u.recorded_share||0))}%"></i></span><span>${pct(u.recorded_share)} <span class="mut">(${f(u.sessions)})</span></span></summary><div class="meta">${esc(u.id)} · ${f(u.direct)} direct GLM · ${f(u.needs_review)} needs review</div>${row(u)}${u.workflows.map(w=>`<details class="node"><summary><span>${esc(w.name)}</span><span class="bar"><i style="width:${Math.max(0,Math.min(100,w.recorded_share||0))}%"></i></span><span>${pct(w.recorded_share)} <span class="mut">(${f(w.sessions)})</span></span></summary><div class="meta">${esc(w.id)} · ${f(w.direct)} direct GLM · ${f(w.needs_review)} needs review</div><div>${esc(w.definition||'')}</div>${row(w)}${w.examples.map(e=>`<div class="ex"><code>${esc(e.session_id)}</code> · <span class="tag">${esc(e.source)}</span><br>“${esc(e.excerpt)}”</div>`).join('')}<details><summary>All ${f(w.trace_ids.length)} trace IDs</summary><div class="ids">${w.trace_ids.map(esc).join(' · ')||'No assigned traces'}</div></details></details>`).join('')}</details>`).join('');
const notes=[];
if(V.validation_count!=null)notes.push(`<div class="panel"><b>Holdout interpretation</b><br>F1 scores show agreement with the GLM-labeled holdout, not ground-truth accuracy. The holdout contains ${f(V.validation_count)} sessions; workflow results are especially sensitive to sparse class counts.</div>`);
notes.push(`<div class="panel"><b>Token provenance</b><br>Recorded token totals use source-reported usage. Estimates and unknown-usage calls are separate and are not added to recorded totals.</div>`);
notes.push(`<div class="panel"><b>Review policy</b><br>Local model suggestions use uncalibrated decision margins and remain marked <code>needs_review</code>; they are not automatic approvals.</div>`);
document.getElementById('notes').innerHTML=notes.join('');
document.getElementById('search').addEventListener('input',e=>{const q=e.target.value.toLowerCase();document.querySelectorAll('#tree .node').forEach(d=>{const hit=d.textContent.toLowerCase().includes(q);d.hidden=!!q&&!hit;if(q&&hit)d.open=true})});
document.getElementById('expand').onclick=()=>document.querySelectorAll('#tree details').forEach(d=>d.open=true);
document.getElementById('collapse').onclick=()=>document.querySelectorAll('#tree details').forEach(d=>d.open=false);
</script></body></html>'''


def _combine(groups):
    keys=("calls","sessions","recorded_input_tokens","recorded_output_tokens","recorded_total_tokens",
          "estimated_total_tokens","unknown_usage_calls")
    result={k:sum(int(g.get(k,0) or 0) for g in groups) for k in keys}
    return {"calls":result["calls"],"sessions":result["sessions"],
            "recorded_input_tokens":result["recorded_input_tokens"],
            "recorded_output_tokens":result["recorded_output_tokens"],
            "recorded_total_tokens":result["recorded_total_tokens"],
            "estimated_total_tokens":result["estimated_total_tokens"],
            "unknown_calls":result["unknown_usage_calls"]}


def hierarchy_html(taxonomy,session_assignments,token_report,path,semantic_metrics=None,
                   input_name="traces.parquet",validation=None,run_summary=None):
    """Write a searchable, offline hierarchy view with reference-style summaries."""
    from collections import Counter
    totals=token_report.get("totals",{})
    group_map={(x.get("use_case_id"),x.get("workflow_id")):x for x in token_report.get("groups",[])}
    assigned=Counter((x.get("use_case_id"),x.get("workflow_id")) for x in session_assignments)
    sources=Counter(x.get("classification_source") for x in session_assignments)
    source_by=Counter((x.get("use_case_id"),x.get("workflow_id"),x.get("classification_source")) for x in session_assignments)
    status_by=Counter((x.get("use_case_id"),x.get("workflow_id"),x.get("status")) for x in session_assignments)
    trace_by={}
    for row in session_assignments:
        trace_by.setdefault((row.get("use_case_id"),row.get("workflow_id")),[]).append(row)
    grand_recorded=max(1,int(totals.get("recorded_total_tokens",0) or 0))
    usecases=[]
    for u in taxonomy.get("business_use_cases",[]):
        uid=u.get("use_case_id")
        wfrows=[]
        for w in taxonomy.get("workflows",[]):
            if w.get("use_case_id")!=uid: continue
            wid=w.get("workflow_id"); m=group_map.get((uid,wid),{})
            rows=trace_by.get((uid,wid),[])
            examples=[]
            for a in rows:
                excerpt=a.get("example_excerpt") or ""
                if a.get("classification_source")=="glm_direct" and excerpt:
                    examples.append({"session_id":a.get("session_id"),"source":"direct GLM","excerpt":excerpt[:300]})
                if len(examples)>=3: break
            wfrows.append({"id":wid,"name":w.get("name") or wid,"definition":w.get("definition","") or "",
                           **_combine([m]),"recorded_share":100*float(m.get("recorded_token_usage_share",0) or 0),
                           "direct":source_by[(uid,wid,"glm_direct")],"needs_review":status_by[(uid,wid,"needs_review")],
                           "examples":examples,"trace_ids":sorted(str(x.get("session_id")) for x in rows)})
        wfrows.sort(key=lambda x:(-x["recorded_total_tokens"],x["id"]))
        groups=[group_map.get((uid,w["id"]),{}) for w in wfrows]
        m=_combine(groups); rows=[x for x in session_assignments if x.get("use_case_id")==uid]
        usecases.append({"id":uid,"name":u.get("name") or uid,"direct":sum(x.get("classification_source")=="glm_direct" for x in rows),
                         "needs_review":sum(x.get("status")=="needs_review" for x in rows),
                         "recorded_share":100*m["recorded_total_tokens"]/grand_recorded,"workflows":wfrows,**m})
    usecases.sort(key=lambda x:(-x["recorded_total_tokens"],x["id"]))
    checks=validation or {}
    vm=semantic_metrics or {}
    validation_rows=vm.get("validation_assignments",[])
    use_correct=sum(x.get("gold_use_case_id")==x.get("predicted_use_case_id") for x in validation_rows)
    wf_correct=sum(x.get("gold_workflow_id")==x.get("predicted_workflow_id") for x in validation_rows)
    direct=int(sources.get("glm_direct",0)); local=int(sources.get("local_ml",0))
    run_summary=run_summary or {}
    summary={"sessions":int(totals.get("sessions",len(session_assignments))),"calls":int(totals.get("calls",0)),
             "recorded_input_tokens":int(totals.get("recorded_input_tokens",0)),
             "recorded_output_tokens":int(totals.get("recorded_output_tokens",0)),
             "recorded_total_tokens":int(totals.get("recorded_total_tokens",0)),
             "estimated_total_tokens":int(totals.get("estimated_total_tokens",0)),
             "unknown_calls":int(totals.get("unknown_usage_calls",0)),"glm_direct":direct,"local_suggestions":local,
             "workflow_count":len(taxonomy.get("workflows",[])),"use_case_count":len(taxonomy.get("business_use_cases",[])),
             "archetypes":int(run_summary.get("archetypes",0)),"selected_archetypes":int(run_summary.get("selected_archetypes",0))}
    data={"input_name":run_summary.get("input_path",input_name),"summary":summary,"business_use_cases":usecases,
          "validation":{"validation_count":vm.get("validation_count"),"use_case_macro_f1":vm.get("use_case_macro_f1"),
                        "workflow_macro_f1":vm.get("workflow_macro_f1"),"use_case_correct":use_correct,
                        "workflow_correct":wf_correct},"checks_passed":bool(checks.get("passed")),
          "check_count":checks.get("check_count",0)}
    safe=json.dumps(data,ensure_ascii=False,separators=(",",":"))
    safe=safe.replace("<","\\u003c").replace(">","\\u003e").replace("&","\\u0026")
    content=_HTML.replace("__DATA__",safe)
    with open(path,"w",encoding="utf-8") as f: f.write(content)
