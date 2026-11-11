"""Loss-aware trace compression for annotation and local feature views."""
import json
import re

def _user_task(text):
    """Extract the actual first user request from wrapped environment prompts, if present."""
    marker="First Observation:"
    if marker in text and "message=" in text:
        tail=text[text.find(marker)+len(marker):]
        m=re.search(r"message=(['\"])",tail)
        if m:
            body=tail[m.end():]
            ends=[p for p in (body.find("\nYou are a customer"),body.find("\nYou are an assistant"),body.find("\nAction:")) if p>=0]
            if ends: body=body[:min(ends)]
            return body.strip().rstrip("'\"")
    # Harnesses often append assistant narration and reminders inside a user-role
    # envelope. Keep the user's request and constraints; discard that appended analysis.
    cuts=[]
    for marker in ("<system-reminder>","Observation: Execution logs:"," The user is asking",\
                   " The user wants", " The user requested", " According to the policy,",\
                   " According to my", " I need to authenticate the user",\
                   "\nYou are a customer service agent", "\nYou are an assistant"):
        p=text.find(marker)
        if p>0: cuts.append(p)
    if cuts: text=text[:min(cuts)]
    # Extract explicit supervisor tasks when a harness wrapper precedes them.
    if "Task from supervisor:" in text:
        p=text.find("Task from supervisor:")
        task=text[p+len("Task from supervisor:"):]
        stop=task.find("Task from supervisor:")
        if stop>0: task=task[:stop]
        return task.strip()
    if text.startswith("Context: "):
        # Context-only prefixes have no user intent until the request marker appears.
        for marker in ("Complete this task", "First Observation:", "Task from supervisor:"):
            p=text.find(marker)
            if p>=0:
                suffix=text[p:]
                return _user_task(suffix) if marker=="First Observation:" else suffix[:800]
    return text.strip()

def _texts(value):
    out=[]
    if isinstance(value, str): return [value]
    if isinstance(value, list):
        for x in value: out.extend(_texts(x))
    elif isinstance(value, dict):
        for k,v in value.items():
            if k in ("content", "text", "value", "parts", "messages"):
                out.extend(_texts(v))
    return [s for s in out if s and s.strip()]

def messages(value):
    if isinstance(value, str):
        try: value=json.loads(value)
        except ValueError: return [{"role":"unknown", "text": value}]
    if isinstance(value, dict): value=value.get("messages", [value])
    result=[]
    for m in value or []:
        if not isinstance(m,dict): continue
        role=str(m.get("role",m.get("type","unknown")))
        result.append({"role":role,"text":" ".join(_texts(m)),"raw":m})
    return result

def _tools(value):
    found=[]
    if isinstance(value, str):
        try: value=json.loads(value)
        except ValueError: return found
    def walk(x):
        if isinstance(x,dict):
            typ=x.get("type")
            if typ in ("tool_call","function_call") or x.get("name") and ("arguments" in x or typ=="function"):
                args=x.get("arguments",{})
                if isinstance(args,str):
                    try: args=json.loads(args)
                    except ValueError: args={}
                found.append({"name":str(x.get("name")),"argument_keys":sorted(args) if isinstance(args,dict) else []})
            for v in x.values(): walk(v)
        elif isinstance(x,list):
            for v in x: walk(v)
    walk(value)
    return found

def session_views(session):
    calls=sorted(session.get("calls",[]),key=lambda x:(x.get("call_index",0) or 0))
    all_messages=[]; tool_calls=[]; tool_definition_names=[]
    # OTel records commonly repeat the full conversation on every call. Parse the first
    # and final input snapshots only; parse output messages for each call.
    input_indexes={0,len(calls)-1} if calls else set()
    for pos,c in enumerate(calls):
        parse_input=pos in input_indexes
        ins=messages(c.get("input_messages")) if parse_input else []
        outs=messages(c.get("output_messages"))
        all_messages.extend(ins+outs)
        tool_calls.extend(_tools(c.get("output_messages")))
        if c is calls[0]:
            for t in _tools(c.get("tool_definitions")):
                if t["name"]: tool_definition_names.append(t["name"])
    raw_users=[m["text"] for m in all_messages if m["role"] in ("user","human") and m["text"]]
    users=[]
    for raw in raw_users:
        task=_user_task(raw)
        if task and (not users or task!=users[-1]): users.append(task[:700])
    asst=[m["text"] for m in all_messages if m["role"] in ("assistant","ai") and m["text"]]
    system=" ".join(m["text"] for m in all_messages if m["role"]=="system")
    if not system and raw_users and "Context: " in raw_users[0]:
        system=raw_users[0].split("Context: ",1)[1][:200]
    first=users[0] if users else ""
    tail=users[-3:] if users else []
    model_ids=sorted({str(c.get("model")) for c in calls if c.get("model")})
    agent_signatures=set()
    for c in calls:
        extra=c.get("extra",{})
        if isinstance(extra,str):
            try: extra=json.loads(extra)
            except ValueError: extra={}
        if isinstance(extra,dict):
            for k in ("agent","agent_name","harness","framework","service.name"):
                if extra.get(k): agent_signatures.add(str(extra[k]))
    metadata={"model":model_ids,"agent":sorted(agent_signatures),"call_count":len(calls)}
    # Preserve negation, dates, quantities, names, conditions, and correction language; only truncate at limits.
    glm={"session_id":session.get("trace_id",session.get("session_id")),"system_excerpt":system[:140],
         "first_user_request":first[:280],"later_user_turns":[x[:170] for x in tail[1:]],
         "assistant_actions":[x[:180] for x in asst[-3:]],"tools":tool_calls[:30],
         "model":metadata["model"],"agent":metadata["agent"],"session_length":len(calls)}
    local=" ".join([first,*tail[1:],*asst[-3:],*(t["name"] for t in tool_calls)])
    intent=" ".join([first,*tail[1:]])
    structural={"system_prefix":re.sub(r"\s+"," ",system[:200]).strip(),
                "tool_names":sorted(set(tool_definition_names)),"model":metadata["model"],
                "tool_vocabulary":sorted({x["name"] for x in tool_calls}),
                "agent_signature":metadata["agent"]}
    return {"glm":glm,"local_text":local,"user_intent":intent,"structural":structural,
            "examples":{"first_user_request":first[:220],"last_assistant":asst[-1][:180] if asst else ""}}
