import time
import logging
from .taxonomy_discovery import classify_batch
log=logging.getLogger("v07.annotations")

def annotate_resumable(client, sessions, taxonomy, checkpoint, deadline, batch_size=8):
    """Sequential GLM-first labeling; adaptively split incomplete batches 8→4→2."""
    parents={w.get("workflow_id"):w.get("use_case_id") for w in taxonomy.get("workflows",[])}
    valid_use_cases={u.get("use_case_id") for u in taxonomy.get("business_use_cases",[])}
    selected_ids={s.get("trace_id") for s in sessions}
    done={}
    for row in checkpoint.load():
        sid=row.get("session_id")
        if sid in selected_ids and parents.get(row.get("workflow_id"))==row.get("use_case_id") and row.get("use_case_id") in valid_use_cases:
            done[sid]=row
    by_id={s["trace_id"]:s for s in sessions}
    for sid,row in done.items():
        if sid in by_id:
            row.setdefault("local_text",by_id[sid].get("local_text","")); row.setdefault("archetype_id",by_id[sid].get("archetype_id"))
            row.setdefault("user_intent",by_id[sid].get("user_intent",""))
    todo=[s for s in sessions if s["trace_id"] not in done]
    queue=[todo[i:i+batch_size] for i in range(0,len(todo),batch_size)]
    while queue and time.monotonic()<deadline:
        batch=queue.pop(0)
        try:
            rows=classify_batch(client,batch,taxonomy)
            for row in rows:
                row["source"]="glm_direct"; row["confidence_type"]="model_reported"
                session=by_id[row["session_id"]]
                row["local_text"]=session.get("local_text",""); row["archetype_id"]=session.get("archetype_id")
                row["user_intent"]=session.get("user_intent","")
                done[row["session_id"]]=row
            checkpoint.save(list(done.values()))
        except Exception:
            log.warning("GLM batch unresolved sessions=%d; will split/retry unresolved subset",len(batch))
            if len(batch)>2 and time.monotonic()<deadline:
                mid=len(batch)//2; queue[0:0]=[batch[:mid],batch[mid:]]
            # At size 1/2, leave unresolved for later resume.
    return list(done.values()),sum(1 for s in sessions if s["trace_id"] not in done)

class JSONCheckpoint:
    def __init__(self,path): self.path=path
    def load(self):
        import json, os
        if not os.path.exists(self.path): return []
        with open(self.path,encoding="utf-8") as f: return json.load(f)
    def save(self,rows):
        import json, os, tempfile
        os.makedirs(os.path.dirname(self.path) or ".",exist_ok=True)
        fd,tmp=tempfile.mkstemp(dir=os.path.dirname(self.path) or ".",prefix=".checkpoint-")
        try:
            with os.fdopen(fd,"w",encoding="utf-8") as f: json.dump(rows,f,ensure_ascii=False)
            os.replace(tmp,self.path)
        finally:
            if os.path.exists(tmp): os.unlink(tmp)
