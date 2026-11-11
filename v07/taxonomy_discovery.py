import json
from .glm_client import IncompleteResponse

DISCOVERY_SYSTEM="""Identify business objectives from trace summaries, independent of model, tool, or agent framework. Return compact JSON only with keys business_use_cases and workflows. Each workflow has a stable lowercase snake_case id, parent use_case_id, one-sentence definition, boundary, and example_session_ids. Taxonomies classify the user's requested end goal. Do not create workflows for authentication, tool use, system setup steps, policy checks, or escalation alone unless that is the actual user goal. Merge equivalent workflows across apps or harnesses. Include one appropriately bounded other workflow under every business use case; needs_review is never a use case. Propose no fixed number of categories."""
CLASSIFY_SYSTEM="""Classify each session by the user's requested end goal, not an agent's first action, authentication step, tool call, policy narration, or harness. Preserve distinctions in the actual user request, including negation, quantities, dates, and corrections. Return compact JSON only: {\"assignments\":[{\"session_id\":string,\"use_case_id\":string,\"workflow_id\":string,\"confidence\":number,\"novelty\":boolean}],\"new_workflow_candidates\":[]}. Use only provided taxonomy IDs unless reporting a candidate. No explanations or chain of thought."""

def discover(client, reps):
    # Small discovery batches avoid truncation; each proposal is compacted before merging.
    proposals=[]
    def compact(x):
        return {"business_use_cases":[{k:u.get(k) for k in ("use_case_id","name","definition")} for u in x.get("business_use_cases",[])],
                "workflows":[{k:w.get(k) for k in ("workflow_id","use_case_id","definition","boundary","example_session_ids")} for w in x.get("workflows",[])]}
    def discover_part(rows):
        payload=[s["glm"] for s in rows]
        try:
            return client.complete(DISCOVERY_SYSTEM,json.dumps({"sessions":payload},ensure_ascii=False),max_tokens=2200)
        except IncompleteResponse:
            if len(rows)<=1: raise
            mid=len(rows)//2
            left=discover_part(rows[:mid]); right=discover_part(rows[mid:])
            return reconcile(left,right)
    def reconcile(left,right):
        user={"candidate_taxonomies":[compact(left),compact(right)],
              "instruction":"Merge duplicate or overlapping business use cases and workflows. Return JSON with business_use_cases and workflows. Keep each definition to one concise sentence. Omit examples, boundaries, and explanations. Preserve distinct workflows."}
        try:
            return client.complete(DISCOVERY_SYSTEM,json.dumps(user,ensure_ascii=False,separators=(",",":")),max_tokens=6000)
        except IncompleteResponse:
            # A compact merge retry keeps the input/output bounded. If even that is
            # incomplete, retain both validated proposal sets for later human review.
            smaller={"use_cases":user["candidate_taxonomies"][0]["business_use_cases"]+user["candidate_taxonomies"][1]["business_use_cases"],
                     "workflows":user["candidate_taxonomies"][0]["workflows"]+user["candidate_taxonomies"][1]["workflows"],
                     "instruction":"Deduplicate IDs; compact JSON only. Keep definitions under 100 characters."}
            try:
                return client.complete(DISCOVERY_SYSTEM,json.dumps(smaller,ensure_ascii=False,separators=(",",":")),max_tokens=6000)
            except IncompleteResponse:
                return _union_taxonomies(left,right)
    for start in range(0,len(reps),8):
        proposals.append(discover_part(reps[start:start+8]))
    if not proposals: return {"business_use_cases":[],"workflows":[]}
    result=compact(proposals[0])
    for proposal in proposals[1:]: result=reconcile(result,proposal)
    return result

def _union_taxonomies(left,right):
    """Deterministic safety fallback that retains category evidence if merge output truncates."""
    uses={}; workflows={}; used_workflow_ids=set()
    for source in (left,right):
        for u in source.get("business_use_cases",[]):
            if u.get("use_case_id"): uses.setdefault(u["use_case_id"],u)
    valid_uses=set(uses)
    for source in (left,right):
        for original in source.get("workflows",[]):
            if not original.get("workflow_id") or original.get("use_case_id") not in valid_uses: continue
            w=dict(original); wid=w["workflow_id"]; parent=w["use_case_id"]
            key=(parent,wid)
            if key in workflows: continue
            if wid in used_workflow_ids:
                wid=f"{parent}__{wid}"
                suffix=2
                while wid in used_workflow_ids:
                    wid=f"{parent}__{original['workflow_id']}__{suffix}"; suffix+=1
                w["workflow_id"]=wid
            used_workflow_ids.add(w["workflow_id"]); workflows[key]=w
    return {"business_use_cases":list(uses.values()),"workflows":list(workflows.values())}

def classify_batch(client, sessions, taxonomy):
    expected={s["glm"]["session_id"] for s in sessions}
    result=client.complete(CLASSIFY_SYSTEM,json.dumps({"taxonomy":taxonomy,"sessions":[s["glm"] for s in sessions]},ensure_ascii=False),max_tokens=1600)
    assignments=result.get("assignments")
    if not isinstance(assignments,list): raise ValueError("missing assignments array")
    got={x.get("session_id") for x in assignments if isinstance(x,dict)}
    if got!=expected: raise ValueError(f"session IDs incomplete/extra: missing={len(expected-got)} extra={len(got-expected)}")
    parents={w["workflow_id"]:w["use_case_id"] for w in taxonomy.get("workflows",[])}
    useids={u["use_case_id"] for u in taxonomy.get("business_use_cases",[])}
    for a in assignments:
        if a.get("workflow_id") not in parents or a.get("use_case_id") not in useids or parents.get(a.get("workflow_id"))!=a.get("use_case_id"): raise ValueError("unapproved taxonomy ID or parent mismatch")
        if not 0<=float(a.get("confidence",-1))<=1: raise ValueError("invalid confidence")
    return assignments
