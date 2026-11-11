from v07.session_builder import build_sessions
from v07.token_analytics import aggregate_tokens

def test_50000_call_synthetic_stress():
    rows=[]
    for i in range(50000):
        s=i//16
        rows.append({"trace_id":f"s{s}","span_id":f"c{i}","call_index":i%16,"model":"m","input_tokens":100,"output_tokens":5,"usage_basis":"source_reported",
          "input_messages":[{"role":"user","parts":[{"type":"text","content":f"Please review task {s}"}]}],
          "output_messages":[{"role":"assistant","content":"done"}],"tool_definitions":[],"extra":{}})
    sessions=build_sessions(rows)
    report=aggregate_tokens(sessions)
    assert len(sessions)==3125 and report["totals"]["calls"]==50000
    assert report["totals"]["recorded_input_tokens"]==5000000
