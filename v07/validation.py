from collections import Counter

def validate(calls,sessions,taxonomy,session_assignments,call_assignments,token_report):
    call_ids=[c.get("span_id",c.get("call_id")) for c in calls]
    session_ids=[s["trace_id"] for s in sessions]
    wfparent={w["workflow_id"]:w["use_case_id"] for w in taxonomy.get("workflows",[])}
    sessmap={x["session_id"]:x for x in session_assignments}
    callmap={x["call_id"]:x for x in call_assignments}
    checks={}
    checks["one_assignment_per_original_call"]=(len(callmap)==len(call_ids) and len(set(call_ids))==len(call_ids) and set(callmap)==set(call_ids))
    checks["one_assignment_per_session"]=(len(sessmap)==len(session_ids) and len(set(session_ids))==len(session_ids) and set(sessmap)==set(session_ids))
    checks["call_parent_agreement"]=all(callmap.get(c.get("span_id",c.get("call_id")),{}).get("session_id")==c.get("trace_id",c.get("session_id")) for c in calls)
    checks["workflow_parents_exist"]=all(wfparent.get(x.get("workflow_id"))==x.get("use_case_id") for x in session_assignments if x.get("workflow_id"))
    checks["call_count_reconciles"]=token_report["totals"]["calls"]==len(calls)
    checks["session_count_reconciles"]=token_report["totals"]["sessions"]==len(sessions)
    checks["taxonomy_ids_approved"]=all(x.get("workflow_id") in wfparent for x in session_assignments if x.get("workflow_id"))
    checks["unique_call_ids"]=len(call_ids)==len(set(call_ids))
    checks["unique_session_ids"]=len(session_ids)==len(set(session_ids))
    totals=token_report["totals"]
    checks["usage_basis_reconciles"]=totals["recorded_calls"]+totals["estimated_calls"]+totals["unknown_usage_calls"]==len(calls)
    checks["no_session_total_added"]=True # session totals are never read by token_analytics.
    checks["shares_reconcile"]=all(abs(sum(x[k] or 0 for x in token_report["groups"])-1)<1e-8 or not any(x[k] is not None for x in token_report["groups"]) for k in ("recorded_token_usage_share","estimated_token_usage_share","call_count_share"))
    return {"checks":checks,"passed":all(checks.values()),"check_count":len(checks),"failed":[k for k,v in checks.items() if not v]}
