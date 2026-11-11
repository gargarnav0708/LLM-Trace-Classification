import json, os, random, time, re
from collections import Counter
from .config import Config
from .loader import iter_calls
from .session_builder import build_sessions
from .archetypes import discover_archetypes, representative_sessions
from .glm_client import GLMClient
from .taxonomy_discovery import discover
from .annotation_manager import annotate_resumable, JSONCheckpoint
from .token_analytics import aggregate_tokens
from .validation import validate
from .visualization import hierarchy_html
from .classifier import HierarchicalClassifier, compare_baselines, compare_text_channels
from .taxonomy_quality import audit_taxonomy
from .active_learning import rank_residual_sessions

def _taxonomy_shape(raw):
    taxonomy={"business_use_cases":raw.get("business_use_cases",[]),"workflows":raw.get("workflows",[])}
    uses={u.get("use_case_id") for u in taxonomy["business_use_cases"]}
    seen=set()
    for u in taxonomy["business_use_cases"]:
        if not u.get("use_case_id") or u["use_case_id"] in seen: raise ValueError("invalid/duplicate use case ID")
        seen.add(u["use_case_id"])
    seen=set()
    for w in taxonomy["workflows"]:
        if not w.get("workflow_id") or w["workflow_id"] in seen or w.get("use_case_id") not in uses: raise ValueError("invalid workflow ID or parent")
        seen.add(w["workflow_id"])
    # Add the required parent-specific fallback without inventing new business use cases.
    for uid in sorted(uses):
        has_other=any(w.get("use_case_id")==uid and "other" in (str(w.get("workflow_id",""))+" "+str(w.get("name",""))).lower() for w in taxonomy["workflows"])
        if has_other: continue
        base=re.sub(r"[^a-z0-9]+","_",str(uid).lower()).strip("_") or "use_case"
        wid=f"{base}_other"; suffix=2
        while wid in seen:
            wid=f"{base}_other_{suffix}"; suffix+=1
        seen.add(wid)
        taxonomy["workflows"].append({"workflow_id":wid,"use_case_id":uid,"name":"Other","definition":"Other user-requested goals in this business use case that do not fit the named workflows.","boundary":"Use only when no named workflow fits; do not use for missing details or review status.","taxonomy_source":"standard_fallback"})
    return taxonomy

def select_direct_sessions(sessions, archetypes, budget=300,seed=7):
    archetypes=[a for a in archetypes if a.get("selected_rank") is not None]
    rng=random.Random(seed); picked=[]; selected=set()
    # A diverse first pass across structural configurations, followed by frequency-weighted rounds.
    for a in archetypes:
        rows=sorted(a["sessions"],key=lambda s:s["trace_id"]); rng.shuffle(rows)
        if rows and len(picked)<budget: picked.append(rows[0]); selected.add(rows[0]["trace_id"])
    groups=[a["sessions"] for a in archetypes]
    while len(picked)<budget:
        progress=False
        for rows in groups:
            choices=[s for s in rows if s["trace_id"] not in selected]
            if choices and len(picked)<budget:
                s=rng.choice(choices); picked.append(s); selected.add(s["trace_id"]); progress=True
        if not progress: break
    return picked

def _semantic_metrics(train_labels, validation_labels, val_sessions, taxonomy, seed=7):
    if not train_labels or not validation_labels:
        return {"status":"not_run; insufficient GLM holdout labels","use_case_macro_f1":None,"workflow_macro_f1":None}
    model=HierarchicalClassifier(seed).fit(train_labels)
    session_by_id={s["trace_id"]:s for s in val_sessions}
    actual=[]; predicted=[]; wf_actual=[]; wf_pred=[]; confidence=[]; correctness=[]; details=[]
    for gold in validation_labels:
        pred=model.predict_one(session_by_id[gold["session_id"]])
        pu=pred.get("use_case_id") or "needs_review"; pw=pred.get("workflow_id") or "needs_review"
        actual.append(gold["use_case_id"]); predicted.append(pu)
        wf_actual.append(gold["workflow_id"]); wf_pred.append(pw)
        if pred.get("confidence") is not None:
            confidence.append(float(pred["confidence"])); correctness.append(float(pu==gold["use_case_id"] and pw==gold["workflow_id"]))
        details.append({"session_id":gold["session_id"],"gold_use_case_id":gold["use_case_id"],"gold_workflow_id":gold["workflow_id"],"predicted_use_case_id":pred.get("use_case_id"),"predicted_workflow_id":pred.get("workflow_id"),"confidence":pred.get("confidence"),"decision_score":pred.get("decision_score"),"confidence_type":pred.get("confidence_type"),"review_status":pred.get("status","needs_review")})
    try:
        from sklearn.metrics import f1_score, precision_recall_fscore_support, confusion_matrix
        uc_f1=float(f1_score(actual,predicted,average="macro",zero_division=0)); wf_f1=float(f1_score(wf_actual,wf_pred,average="macro",zero_division=0))
        weighted=float(f1_score(actual,predicted,average="weighted",zero_division=0))
        labs=sorted(set(wf_actual)|set(wf_pred)); pr,re,_,_=precision_recall_fscore_support(wf_actual,wf_pred,labels=labs,zero_division=0)
        per={label:{"precision":float(pr[i]),"recall":float(re[i])} for i,label in enumerate(labs)}
        cm={"labels":labs,"matrix":confusion_matrix(wf_actual,wf_pred,labels=labs).tolist()}
        brier=sum((a-b)**2 for a,b in zip(confidence,correctness))/len(confidence) if len(confidence)>=20 else None
    except Exception as e:
        uc_f1=wf_f1=weighted=None; per={}; cm=None; brier=None
    baseline=compare_baselines([HierarchicalClassifier._text(x) for x in train_labels],[x["use_case_id"] for x in train_labels],[x.get("archetype_id",x["session_id"]) for x in train_labels],seed) if len(train_labels)>=10 else {}
    ablation=compare_text_channels(train_labels) if len(train_labels)>=10 else {}
    return {"status":"GLM_pseudo_label_agreement_not_ground_truth","validation_count":len(validation_labels),"use_case_macro_f1":uc_f1,"workflow_macro_f1":wf_f1,"weighted_f1":weighted,"per_workflow_precision_recall":per,"confusion_matrix":cm,"brier_score":None,"calibration":"LinearSVC decision margins are uncalibrated; no probability or Brier score claimed","glm_model_agreement":None,"selected_residual_model":"class-balanced LinearSVC with word unigram/bigram TF-IDF","selected_text_view":"user intent duplicated once plus short conversation context","grouped_training_cv_baselines":baseline,"grouped_feature_ablation":ablation,"validation_assignments":details}

def create_assignments(sessions,labels,taxonomy):
    labelmap={x["session_id"]:x for x in labels}; defs={x["workflow_id"]:x for x in taxonomy["workflows"]}
    rows=[]
    for s in sessions:
        x=labelmap.get(s["trace_id"])
        if x:
            d=defs[x["workflow_id"]]
            rows.append({"session_id":s["trace_id"],"use_case_id":x["use_case_id"],"workflow_id":x["workflow_id"],"classification_source":"glm_direct","confidence":x.get("confidence"),"confidence_type":"model_reported","decision_score":None,"status":"classified","example_excerpt":s["examples"]["first_user_request"][:180]})
        else:
            rows.append({"session_id":s["trace_id"],"use_case_id":None,"workflow_id":None,"classification_source":"unresolved","confidence":None,"confidence_type":"unavailable","decision_score":None,"status":"needs_review","example_excerpt":""})
    return rows

def _safe_propagation(sessions,direct_labels,threshold=.82):
    """Propagate only unanimous, multi-example archetypes with close user-intent similarity."""
    by_id={s["trace_id"]:s for s in sessions}; by_arch={}
    for lab in direct_labels:
        s=by_id.get(lab["session_id"])
        if s: by_arch.setdefault(s["archetype_id"],[]).append(lab)
    propagated={}
    for aid,labs in by_arch.items():
        pairs={(x["use_case_id"],x["workflow_id"]) for x in labs}
        if len(labs)<3 or len(pairs)!=1: continue
        candidates=[s for s in sessions if s["archetype_id"]==aid and s["trace_id"] not in {x["session_id"] for x in direct_labels}]
        reps=[by_id[x["session_id"]].get("user_intent","") for x in labs]
        if not candidates: continue
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer
            from sklearn.metrics.pairwise import cosine_similarity
            texts=reps+[s.get("user_intent","") for s in candidates]
            matrix=TfidfVectorizer(ngram_range=(1,2),min_df=1).fit_transform(texts)
            centroid=matrix[:len(reps)].mean(axis=0)
            sims=cosine_similarity(matrix[len(reps):],centroid).ravel()
        except Exception: continue
        uc,wf=next(iter(pairs))
        for s,sim in zip(candidates,sims):
            if float(sim)>=threshold:
                propagated[s["trace_id"]]={"use_case_id":uc,"workflow_id":wf,"source":"structural_propagation","confidence":float(sim),"confidence_type":"semantic_similarity_score","status":"classified"}
    return propagated

def _residual_predictions(sessions,direct_ids,propagated,train_labels,seed):
    model=HierarchicalClassifier(seed).fit(train_labels)
    out={}
    for s in sessions:
        if s["trace_id"] in direct_ids or s["trace_id"] in propagated: continue
        pred=model.predict_one(s)
        if pred.get("use_case_id"):
            out[s["trace_id"]]={"use_case_id":pred["use_case_id"],"workflow_id":pred.get("workflow_id"),"source":"local_ml","confidence":None,"decision_score":pred.get("decision_score"),"confidence_type":"uncalibrated_margin","status":"needs_review"}
    return out

def run(input_path,output_dir,live_glm=False,budget=300,max_archetypes=30,overwrite=False,taxonomy_path=None,annotations_dir=None,residual_budget=10000):
    started=time.monotonic()
    os.makedirs(output_dir,exist_ok=True)
    expected=("results.json","taxonomy.json","trace_assignments.json","session_assignments.json","validation_report.json","insights.json","hierarchy.html","diagnostic_report.json","taxonomy_audit.json")
    conflicts=[name for name in expected if os.path.exists(os.path.join(output_dir,name))]
    if conflicts and not overwrite: raise FileExistsError(f"Output exists ({conflicts[0]}); choose a new directory or pass --overwrite")
    sessions=build_sessions(iter_calls(input_path))
    calls=[c for s in sessions for c in s["calls"]]
    archetypes=discover_archetypes(sessions,max_archetypes)
    taxonomy={"business_use_cases":[],"workflows":[]}; labels=[]; validation_labels=[]; residual_labels=[]; pending=0; semantic={"status":"analysis_only; no GLM labels"}
    if live_glm or taxonomy_path or annotations_dir:
        if taxonomy_path:
            with open(taxonomy_path,encoding="utf-8") as f: taxonomy=_taxonomy_shape(json.load(f))
        elif live_glm:
            pass
        else:
            raise ValueError("Cached labels require --taxonomy-file")
        selected_arch=[a for a in archetypes if a.get("selected_rank") is not None]
        train_arch=[a for a in selected_arch if a["selected_rank"]<=max(1,int(len(selected_arch)*.8))]
        holdout_arch=[a for a in selected_arch if a["selected_rank"]>max(1,int(len(selected_arch)*.8))]
        train_sessions=select_direct_sessions(sessions,train_arch,budget,7)
        validation_sessions=select_direct_sessions(sessions,holdout_arch,50,8)
        if annotations_dir:
            def hydrate(filename,selected):
                p=os.path.join(annotations_dir,filename)
                with open(p,encoding="utf-8") as f: old=json.load(f)
                valid={s["trace_id"]:s for s in selected}
                parents={w["workflow_id"]:w["use_case_id"] for w in taxonomy["workflows"]}
                result=[]
                for x in old:
                    s=valid.get(x.get("session_id"))
                    if not s or parents.get(x.get("workflow_id"))!=x.get("use_case_id"): continue
                    row=dict(x); row.update({"source":"glm_direct","local_text":s["local_text"],"user_intent":s.get("user_intent",""),"archetype_id":s["archetype_id"]})
                    result.append(row)
                return result
            labels=hydrate("annotations_train.json",sessions)
            validation_labels=hydrate("annotations_validation.json",sessions)
            if {x["session_id"] for x in labels} & {x["session_id"] for x in validation_labels}:
                raise ValueError("Cached training and validation annotation IDs overlap")
            session_index={s["trace_id"]:s for s in sessions}
            validation_sessions=[session_index[x["session_id"]] for x in validation_labels]
            pending=max(0,300-len(labels))+max(0,50-len(validation_labels))
            semantic=_semantic_metrics(labels,validation_labels,validation_sessions,taxonomy,7)
        elif live_glm:
            cfg=Config.from_env()
            if not cfg.api_key: raise RuntimeError("Live GLM mode requires HYDE_API_KEY")
            client=GLMClient(cfg.api_key,cfg.api_base,cfg.model,os.path.join(output_dir,"..","work","glm_cache.sqlite"),cfg.request_timeout)
            if not taxonomy_path:
                reps=representative_sessions(selected_arch,2)
                taxonomy=_taxonomy_shape(discover(client,reps))
            deadline=started+cfg.annotation_cutoff_minutes*60
            checkpoint_dir=os.path.join(output_dir,"..","work")
            labels,pending_train=annotate_resumable(client,train_sessions,taxonomy,JSONCheckpoint(os.path.join(checkpoint_dir,"annotations_train.json")),deadline,cfg.batch_size)
            if time.monotonic()<deadline:
                validation_labels,pending_val=annotate_resumable(client,validation_sessions,taxonomy,JSONCheckpoint(os.path.join(checkpoint_dir,"annotations_validation.json")),deadline,cfg.batch_size)
            else: pending_val=len(validation_sessions)
            pending=pending_train+pending_val
            semantic=_semantic_metrics(labels,validation_labels,validation_sessions,taxonomy,cfg.random_seed)
            # Keep the independent holdout isolated. Spend remaining wall time on
            # uncertain, novel, rare-workflow, and disagreement cases under the
            # exact saved/discovered taxonomy.
            if time.monotonic()<deadline and labels:
                excluded={s["trace_id"] for s in train_sessions+validation_sessions}
                candidates=[s for s in sessions if s["trace_id"] not in excluded]
                model=HierarchicalClassifier(cfg.random_seed).fit(labels)
                ranked=rank_residual_sessions(candidates,labels,model,cfg.random_seed)
                residual_sessions=[x["session"] for x in ranked[:max(0,int(residual_budget))]]
                residual_labels,pending_residual=annotate_resumable(
                    client,residual_sessions,taxonomy,
                    JSONCheckpoint(os.path.join(checkpoint_dir,"annotations_residual.json")),
                    deadline,cfg.batch_size)
                pending+=pending_residual
    final_labels=labels+validation_labels+residual_labels
    assignments=create_assignments(sessions,final_labels,taxonomy)
    direct_ids={x["session_id"] for x in final_labels}
    propagated=_safe_propagation(sessions,labels+residual_labels)
    residual=_residual_predictions(sessions,direct_ids,propagated,labels+residual_labels,7) if labels else {}
    for row in assignments:
        sid=row["session_id"]
        extra=propagated.get(sid) or residual.get(sid)
        if extra:
            row.update({"use_case_id":extra["use_case_id"],"workflow_id":extra["workflow_id"],"classification_source":extra["source"],"confidence":extra["confidence"],"decision_score":extra.get("decision_score",extra.get("confidence")),"confidence_type":extra["confidence_type"],"status":extra["status"]})
    assignment_map={x["session_id"]:x for x in assignments}
    token_report=aggregate_tokens(sessions,assignment_map)
    call_assignments=[]
    for s in sessions:
        a=assignment_map[s["trace_id"]]
        for c in s["calls"]:
            call_assignments.append({"call_id":c.get("span_id"),"session_id":s["trace_id"],"classification_source":a["classification_source"],"use_case_id":a["use_case_id"],"workflow_id":a["workflow_id"],"confidence":a["confidence"],"confidence_type":a["confidence_type"],"status":a["status"]})
    validation=validate(calls,sessions,taxonomy,assignments,call_assignments,token_report)
    validation={**validation,"semantic_metrics":semantic}
    report={"input_path":os.path.basename(input_path),"calls":len(calls),"sessions":len(sessions),"archetypes":len(archetypes),"selected_archetypes":sum(a["selected_rank"] is not None for a in archetypes),"direct_glm_training_labels":len(labels),"direct_glm_validation_labels":len(validation_labels),"direct_glm_residual_labels":len(residual_labels),"residual_budget":int(residual_budget),"safely_propagated_sessions":len(propagated),"residual_model_suggestions":len(residual),"unresolved_batches":pending,"execution_mode":"live_glm" if live_glm else ("cached_annotation_replay" if annotations_dir else "analysis_only"),"validation":validation}
    tax_audit=audit_taxonomy(taxonomy,labels+residual_labels,validation_labels)
    outputs={"results.json":{"summary":report,"token_analytics":token_report},"taxonomy.json":taxonomy,"taxonomy_audit.json":tax_audit,"trace_assignments.json":call_assignments,"session_assignments.json":assignments,"validation_report.json":validation,"insights.json":{"measured":[],"hypotheses":[],"note":"Business insights are omitted until independently validated classifications exist."},"diagnostic_report.json":{"summary":report,"archetypes":[{k:v for k,v in a.items() if k!="sessions"} for a in archetypes],"token_totals":token_report["totals"]}}
    for name,obj in outputs.items():
        with open(os.path.join(output_dir,name),"w",encoding="utf-8") as f: json.dump(obj,f,ensure_ascii=False,indent=2)
    hierarchy_html(taxonomy,assignments,token_report,os.path.join(output_dir,"hierarchy.html"),semantic,
                   validation=validation,run_summary=report)
    return report
