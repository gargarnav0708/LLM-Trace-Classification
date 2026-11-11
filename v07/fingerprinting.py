import hashlib
import json

def fingerprint(parts):
    payload=json.dumps(parts,sort_keys=True,ensure_ascii=False,separators=(",",":"))
    return hashlib.sha256(payload.encode()).hexdigest()

def archetype_key(structural):
    return fingerprint({"system_prefix":structural.get("system_prefix",""),
                       "tool_names":sorted(set(structural.get("tool_names",[]))),
                       "model":sorted(structural.get("model",[])),
                       "agent":structural.get("agent_signature","")})
