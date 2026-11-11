from collections import defaultdict

def _add(dst,key,val):
    if val is not None and isinstance(val,(int,float)) and val>=0: dst[key]+=int(val)

def aggregate_tokens(sessions, assignments=None):
    assignments=assignments or {}; by=defaultdict(lambda:{"calls":0,"sessions":set(),"recorded_input":0,"recorded_output":0,"estimated_input":0,"estimated_output":0,"unknown_calls":0,"recorded_calls":0,"estimated_calls":0})
    total={"calls":0,"sessions":set(),"recorded_input":0,"recorded_output":0,"estimated_input":0,"estimated_output":0,"unknown_calls":0,"recorded_calls":0,"estimated_calls":0}
    for s in sessions:
        a=assignments.get(s["trace_id"],{}); key=(a.get("use_case_id","unassigned"),a.get("workflow_id","unassigned"))
        for c in s["calls"]:
            for d in (total,by[key]):
                d["calls"]+=1; d["sessions"].add(s["trace_id"])
            basis=c.get("usage_basis"); inp=c.get("input_tokens"); out=c.get("output_tokens")
            for d in (total,by[key]):
                if basis=="source_reported" and inp is not None and out is not None:
                    _add(d,"recorded_input",inp); _add(d,"recorded_output",out); d["recorded_calls"]+=1
                elif basis=="estimated_cl100k_base" and inp is not None and out is not None:
                    _add(d,"estimated_input",inp); _add(d,"estimated_output",out); d["estimated_calls"]+=1
                else: d["unknown_calls"]+=1
    rec_total=total["recorded_input"]+total["recorded_output"]; est_total=total["estimated_input"]+total["estimated_output"]
    rows=[]
    for (uc,wf),d in sorted(by.items()):
        rec=d["recorded_input"]+d["recorded_output"]; est=d["estimated_input"]+d["estimated_output"]
        rows.append({"use_case_id":uc,"workflow_id":wf,"calls":d["calls"],"sessions":len(d["sessions"]),
          "recorded_input_tokens":d["recorded_input"],"recorded_output_tokens":d["recorded_output"],"recorded_total_tokens":rec,
          "estimated_input_tokens":d["estimated_input"],"estimated_output_tokens":d["estimated_output"],"estimated_total_tokens":est,
          "unknown_usage_calls":d["unknown_calls"],"recorded_token_usage_share":rec/rec_total if rec_total else None,
          "estimated_token_usage_share":est/est_total if est_total else None,"call_count_share":d["calls"]/total["calls"] if total["calls"] else None,
          "average_recorded_tokens_per_recorded_call":rec/d["recorded_calls"] if d["recorded_calls"] else None})
    return {"totals":{"calls":total["calls"],"sessions":len(total["sessions"]),"recorded_calls":total["recorded_calls"],"estimated_calls":total["estimated_calls"],"unknown_usage_calls":total["unknown_calls"],"recorded_input_tokens":total["recorded_input"],"recorded_output_tokens":total["recorded_output"],"recorded_total_tokens":rec_total,"estimated_input_tokens":total["estimated_input"],"estimated_output_tokens":total["estimated_output"],"estimated_total_tokens":est_total},"groups":rows}
