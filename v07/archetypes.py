from collections import defaultdict

def discover_archetypes(sessions, max_selected=30):
    groups=defaultdict(list)
    for s in sessions: groups[s["archetype_id"]].append(s)
    ranked=[]
    for aid,ss in groups.items():
        texts=[s.get("user_intent",s["local_text"]) or "" for s in ss]
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer
            from sklearn.metrics.pairwise import cosine_similarity
            matrix=TfidfVectorizer(ngram_range=(1,2),min_df=1,max_features=10000,sublinear_tf=True).fit_transform(texts)
            cohesion=float(cosine_similarity(matrix,matrix.mean(axis=0)).mean())
            # Pairwise similarity to centroid serves as a stable heterogeneous-group diagnostic.
            import numpy as np
            diversity=float(np.clip(1-cohesion,0,1))
        except Exception:
            tokens=[set(t.lower().split()) for t in texts]
            diversity=len(set.union(*tokens))/max(1,sum(map(len,tokens))) if tokens else 1.0
            cohesion=1-diversity
        n=max(1,len(ss))
        row={"archetype_id":aid,"session_count":n,"intent_cohesion":round(cohesion,4),
             "unique_tool_signatures":len({tuple(s["structural"].get("tool_names",[])) for s in ss}),
             "intent_diversity":round(diversity,4),"potential_business_relevance":"requires semantic review","sessions":ss}
        ranked.append(row)
    ranked.sort(key=lambda x:(-x["session_count"],-x["intent_cohesion"],x["archetype_id"]))
    for i,x in enumerate(ranked): x["selected_rank"]=i+1 if i<max_selected else None
    return ranked

def representative_sessions(archetypes, limit=2):
    picks=[]
    for a in archetypes:
        ss=a["sessions"]
        # Spread picks lexically by intent text for coverage; no same-intent-only assumption.
        ss=sorted(ss,key=lambda s:(s["local_text"].lower(),s["trace_id"]))
        if len(ss)<=limit: picks.extend(ss)
        else:
            ix=sorted({round(i*(len(ss)-1)/(limit-1)) for i in range(limit)}) if limit>1 else [len(ss)//2]
            picks.extend(ss[i] for i in ix)
    return picks
