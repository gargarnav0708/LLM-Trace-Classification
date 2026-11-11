from .compression import session_views
from .fingerprinting import archetype_key

def build_sessions(calls):
    # The supplied Parquet is explicitly grouped by session. Stream one session at a
    # time so repeated full-context snapshots are not retained for the whole dataset.
    sessions=[]; current_sid=None; current=[]; seen_ids=set(); closed=set()
    def finish(sid,rows):
        rows.sort(key=lambda x:(x.get("call_index") if x.get("call_index") is not None else 0))
        s={"trace_id":sid,"calls":rows}
        views=session_views(s); s.update(views)
        s["archetype_id"]=archetype_key(views["structural"])
        # Retain original IDs and token provenance, not complete per-call payloads.
        s["calls"]=[{k:r.get(k) for k in ("trace_id","span_id","call_index","model","input_tokens","output_tokens","usage_basis","extra")} for r in rows]
        return s
    for row in calls:
        sid=row.get("trace_id",row.get("session_id")); cid=row.get("span_id",row.get("call_id"))
        if not sid or not cid: raise ValueError("Each call needs original trace/session and span/call IDs")
        sid=str(sid)
        if cid in seen_ids: raise ValueError(f"Duplicate call ID: {cid}")
        seen_ids.add(cid)
        if current_sid is not None and sid!=current_sid:
            sessions.append(finish(current_sid,current)); closed.add(current_sid); current=[]
            if sid in closed: raise ValueError("Input sessions are interleaved; sort/group calls by trace_id before loading")
        current_sid=sid; current.append(row)
    if current: sessions.append(finish(current_sid,current))
    return sessions
