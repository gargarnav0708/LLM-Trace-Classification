import json
from v07.compression import messages, session_views
from v07.fingerprinting import archetype_key
from v07.session_builder import build_sessions
from v07.token_analytics import aggregate_tokens
from v07.glm_client import GLMClient, IncompleteResponse
from v07.active_learning import rank_candidates, diverse_select
from v07.classifier import HierarchicalClassifier
from v07.pipeline import _safe_propagation, _taxonomy_shape
from v07.taxonomy_discovery import discover
from v07.taxonomy_quality import audit_taxonomy
from v07.compression import _user_task
from v07.annotation_manager import annotate_resumable, JSONCheckpoint
import time

def row(sid="s1", cid="c1", idx=0, inp=10, out=3, basis="source_reported", inmsg=None, outmsg=None):
    return {"trace_id":sid,"span_id":cid,"call_index":idx,"model":"m1","input_tokens":inp,"output_tokens":out,"usage_basis":basis,
      "input_messages":inmsg or [{"role":"user","content":"Please do not cancel order 123 before Friday."}],
      "output_messages":outmsg or [{"role":"assistant","parts":[{"type":"text","content":"I will check."}]}],
      "tool_definitions":[{"type":"function","name":"lookup_order","parameters":{"properties":{"id":{}}}}],"extra":{}}

def test_varying_message_shapes_and_compression_preserve_constraints():
    shapes=[
      [{"role":"user","content":"Don't remove item A-17 unless approved on 2026-11-08."}],
      [{"role":"user","parts":[{"type":"text","content":"Do not remove item A-17."}]}],
      {"messages":[{"role":"user","content":[{"type":"text","text":"Never cancel order 991."}]}]},
    ]
    assert "Don't" in messages(shapes[0])[0]["text"]
    for i,shape in enumerate(shapes):
        views=session_views({"trace_id":str(i),"calls":[row(inmsg=shape)]})
        txt=views["glm"]["first_user_request"]
        assert any(term in txt.lower() for term in ("don't","do not","never"))
    assert "A-17" in session_views({"calls":[row(inmsg=shapes[0]) ]})["local_text"]
    wrapped="Context: {'policy': 'large policy'}\n\nFirst Observation: sender='user' message='Please return order 882 on 2026-11-08.'\nYou are a customer service agent."
    view=session_views({"trace_id":"wrapped","calls":[row(inmsg=[{"role":"user","content":wrapped}]) ]})
    assert view["glm"]["first_user_request"]=="Please return order 882 on 2026-11-08."
    assert _user_task("Please cancel order 12. The user is asking to cancel their order. <system-reminder>ignore") == "Please cancel order 12."

def test_structural_stability_and_call_order():
    calls=[row("s1","c2",1),row("s1","c1",0)]
    s=build_sessions(calls)[0]
    assert [c["span_id"] for c in s["calls"]]==["c1","c2"]
    assert s["archetype_id"]==archetype_key(s["structural"])

def test_token_provenance_and_unknown_not_zero():
    sessions=build_sessions([row("a","1",0,10,2,"source_reported"),row("b","2",0,50,6,"estimated_cl100k_base"),row("c","3",0,None,9,"unknown")])
    t=aggregate_tokens(sessions)
    assert t["totals"]["recorded_input_tokens"]==10 and t["totals"]["recorded_output_tokens"]==2
    assert t["totals"]["estimated_input_tokens"]==50 and t["totals"]["estimated_output_tokens"]==6
    assert t["totals"]["unknown_usage_calls"]==1

def test_glm_json_extraction_rejects_partial_and_active_diversity():
    assert GLMClient.parse_json('prefix {"assignments": []} suffix')=={"assignments":[]}
    try: GLMClient.parse_json('{"assignments":')
    except ValueError: pass
    else: assert False
    rows=rank_candidates([{"session_id":"a","novelty":1},{"session_id":"b","uncertainty":1}])
    assert diverse_select(rows,1)[0]["session_id"]==rows[0]["session_id"]

def test_hierarchical_model_and_consensus_propagation():
    sessions=[]; labels=[]
    intents=["refund my returned purchase", "refund for returned order", "please issue purchase refund",
             "debug python import error", "fix python package import failure", "repair module import bug"]
    raw=[]
    for i,intent in enumerate(intents):
        raw.append(row(f"s{i}",f"c{i}",inmsg=[{"role":"user","content":intent}],outmsg=[{"role":"assistant","content":"I will help."}]))
    sessions=build_sessions(raw)
    for i,s in enumerate(sessions):
        labels.append({"session_id":s["trace_id"],"local_text":s["local_text"],"use_case_id":"commerce" if i<3 else "engineering",
          "workflow_id":"refund" if i<3 else "debugging","source":"glm_direct","archetype_id":s["archetype_id"]})
    # Archetypes differ by task text but share structure; model fit works with direct labels.
    model=HierarchicalClassifier().fit(labels)
    pred=model.predict_one(sessions[0])
    assert pred["use_case_id"] in {"commerce","engineering"}
    assert pred["status"]=="needs_review" and pred["confidence"] is None
    # Consensus does not propagate with fewer than three agreeing direct labels in a group.
    assert _safe_propagation(sessions,labels)=={}

def test_taxonomy_audit_flags_process_steps_and_low_support():
    taxonomy={"business_use_cases":[{"use_case_id":"commerce"}],"workflows":[{"workflow_id":"authenticate_customer_identity","use_case_id":"commerce","definition":"Authenticate the user before account actions."}]}
    audit=audit_taxonomy(taxonomy,[{"workflow_id":"authenticate_customer_identity"}],[])
    assert any(x["type"]=="possible_process_step_as_workflow" for x in audit["flags"])
    assert any(x["type"]=="missing_other_workflow" for x in audit["flags"])
    normalized=_taxonomy_shape(taxonomy)
    assert any(w["use_case_id"]=="commerce" and w.get("taxonomy_source")=="standard_fallback" for w in normalized["workflows"])

def test_resume_cache_filters_sessions_outside_current_split(tmp_path):
    cp=JSONCheckpoint(str(tmp_path/"labels.json"))
    cp.save([{"session_id":"keep","use_case_id":"commerce","workflow_id":"orders","confidence":.9},
             {"session_id":"stale","use_case_id":"commerce","workflow_id":"orders","confidence":.9}])
    taxonomy={"business_use_cases":[{"use_case_id":"commerce"}],"workflows":[{"workflow_id":"orders","use_case_id":"commerce"}]}
    sessions=[{"trace_id":"keep","local_text":"intent","user_intent":"intent","archetype_id":"a"}]
    rows,pending=annotate_resumable(None,sessions,taxonomy,cp,time.monotonic()+1)
    assert [x["session_id"] for x in rows]==["keep"] and pending==0

def test_taxonomy_discovery_recovers_from_truncated_merge():
    class FakeClient:
        def complete(self, system, user, max_tokens):
            payload=json.loads(user)
            if "candidate_taxonomies" in payload:
                raise IncompleteResponse("simulated finish_reason=length")
            if "use_cases" in payload:
                return {"business_use_cases":[{"use_case_id":"commerce","name":"Commerce","definition":"Purchase support."}],
                  "workflows":[{"workflow_id":"orders","use_case_id":"commerce","definition":"Help with orders."}]}
            n=int(payload["sessions"][0]["session_id"])
            return {"business_use_cases":[{"use_case_id":f"use_{n}","name":"Use","definition":"Task."}],
              "workflows":[{"workflow_id":f"wf_{n}","use_case_id":f"use_{n}","definition":"Task."}]}
    reps=[{"glm":{"session_id":str(i)}} for i in range(9)]
    result=discover(FakeClient(),reps)
    assert result["business_use_cases"][0]["use_case_id"]=="commerce"

def test_glm_client_flags_length_finish_reason(tmp_path):
    client=GLMClient("secret","https://example.invalid/v1","test",str(tmp_path/"cache.sqlite"))
    client._request=lambda payload:{"choices":[{"finish_reason":"length","message":{"content":"{}"}}]}
    try: client.complete("system","user")
    except IncompleteResponse: pass
    else: assert False
