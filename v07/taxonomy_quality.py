"""Evidence-only taxonomy diagnostics. Findings never rewrite labels automatically."""
from collections import Counter
import re

PROCESS_STEP=re.compile(r"authenticate|identity verification|verify (?:the )?user|policy check|escalat(?:e|ion)",re.I)

def audit_taxonomy(taxonomy,training_labels=(),validation_labels=()):
    train=Counter(x.get("workflow_id") for x in training_labels)
    valid=Counter(x.get("workflow_id") for x in validation_labels)
    usecases=taxonomy.get("business_use_cases",[]); workflows=taxonomy.get("workflows",[])
    has_other={u.get("use_case_id"):False for u in usecases}
    flags=[]
    for w in workflows:
        wid=w.get("workflow_id",""); parent=w.get("use_case_id")
        if re.search(r"(?:^|[_\s])other(?:$|[_\s])",wid,re.I) or str(w.get("name","")).strip().lower()=="other": has_other[parent]=True
        evidence=f"{wid} {w.get('definition','')}"
        if PROCESS_STEP.search(evidence):
            flags.append({"type":"possible_process_step_as_workflow","workflow_id":wid,"use_case_id":parent,"reason":"Name or definition may describe a prerequisite or agent procedure instead of the user's business objective; review examples."})
        if train[wid]<3 and w.get("taxonomy_source")!="standard_fallback":
            flags.append({"type":"low_direct_training_support","workflow_id":wid,"use_case_id":parent,"direct_training_examples":train[wid],"reason":"Insufficient direct GLM examples for safe residual workflow inference."})
        if 0<valid[wid]<3:
            flags.append({"type":"low_pseudo_validation_support","workflow_id":wid,"use_case_id":parent,"GLM_holdout_examples":valid[wid],"reason":"Agreement estimate is unstable with fewer than three GLM-labelled holdout examples."})
    for u in usecases:
        uid=u.get("use_case_id")
        if not has_other.get(uid,False):
            flags.append({"type":"missing_other_workflow","use_case_id":uid,"reason":"Consider a parent-specific other workflow for genuine tasks outside the named workflows."})
    fallbacks=[w["workflow_id"] for w in workflows if w.get("taxonomy_source")=="standard_fallback"]
    return {"use_case_count":len(usecases),"workflow_count":len(workflows),"direct_training_examples":sum(train.values()),"GLM_holdout_examples":sum(valid.values()),"standard_other_fallbacks_added":fallbacks,"flags":flags,"automatic_changes":[]}
