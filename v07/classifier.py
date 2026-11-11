"""Residual hierarchical sklearn classifier; scores remain uncalibrated by default."""
from collections import defaultdict, Counter

class HierarchicalClassifier:
    def __init__(self, seed=7):
        self.seed=seed; self.use_model=None; self.workflow_models={}; self.workflow_parent={}
    @staticmethod
    def _pipeline(classifier):
        from sklearn.pipeline import Pipeline
        from sklearn.feature_extraction.text import TfidfVectorizer
        # Grouped CV on the saved GLM labels favored word 1–2 grams with LinearSVC
        # over the previous word+character LogisticRegression stack.
        return Pipeline([("features",TfidfVectorizer(ngram_range=(1,2),min_df=1,sublinear_tf=True,max_features=50000)),("clf",classifier)])
    @staticmethod
    def _text(row):
        # Reweight the actual user request relative to assistant narration and tool names.
        # Cross-validation on the GLM-labelled training groups improved modestly at 2x.
        intent=row.get("user_intent") or ""
        context=row.get("local_text","")
        return f"{intent} {intent} {context}".strip() if intent else context
    def fit(self, labels):
        # labels require direct GLM or separately validated propagation only.
        labels=[x for x in labels if x.get("source") in ("glm_direct","structural_propagation") and x.get("use_case_id") and x.get("workflow_id")]
        if len(labels)<2: return self
        from sklearn.svm import LinearSVC
        texts=[self._text(x) for x in labels]; y=[x["use_case_id"] for x in labels]
        self.use_model=self._pipeline(LinearSVC(class_weight="balanced",random_state=self.seed))
        self.use_model.fit(texts,y)
        groups=defaultdict(list)
        for x in labels: groups[x["use_case_id"]].append(x)
        for parent,rows in groups.items():
            counts=Counter(x["workflow_id"] for x in rows)
            rows=[x for x in rows if counts[x["workflow_id"]]>=3]
            distinct={x["workflow_id"] for x in rows}
            if len(rows)<3: continue
            if len(distinct)==1:
                self.workflow_models[parent]=next(iter(distinct)); continue
            model=self._pipeline(LinearSVC(class_weight="balanced",random_state=self.seed))
            model.fit([self._text(x) for x in rows],[x["workflow_id"] for x in rows])
            self.workflow_models[parent]=model
        self.workflow_parent={x["workflow_id"]:x["use_case_id"] for x in labels}
        return self
    def predict_one(self, session):
        if self.use_model is None: return {"status":"needs_review","confidence":None,"confidence_type":"uncalibrated"}
        text=self._text(session); parent=str(self.use_model.predict([text])[0])
        uc_scores=self.use_model.decision_function([text])
        uc_margin=float(abs(uc_scores[0])) if getattr(uc_scores,"ndim",1)==1 else float(max(uc_scores[0]))
        wfmodel=self.workflow_models.get(parent)
        if wfmodel is None: return {"use_case_id":parent,"workflow_id":None,"confidence":None,"decision_score":uc_margin,"confidence_type":"uncalibrated_margin","status":"needs_review"}
        if isinstance(wfmodel,str): return {"use_case_id":parent,"workflow_id":wfmodel,"confidence":None,"decision_score":uc_margin,"confidence_type":"uncalibrated_margin","status":"needs_review"}
        workflow=str(wfmodel.predict([text])[0]); wf_scores=wfmodel.decision_function([text])
        wf_margin=float(abs(wf_scores[0])) if getattr(wf_scores,"ndim",1)==1 else float(max(wf_scores[0]))
        return {"use_case_id":parent,"workflow_id":workflow,"confidence":None,"decision_score":uc_margin,"workflow_decision_score":wf_margin,"confidence_type":"uncalibrated_margin","status":"needs_review"}

def compare_baselines(texts, labels, groups=None, seed=7):
    """Compare LR, LinearSVC, ComplementNB on grouped CV where possible; imports lazily."""
    from sklearn.model_selection import GroupKFold, cross_val_score
    from sklearn.pipeline import make_pipeline
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.svm import LinearSVC
    from sklearn.naive_bayes import ComplementNB
    results={}; n_splits=min(5,len(set(labels)))
    groups=groups or [str(i) for i in range(len(labels))]
    if n_splits<2: return results
    cv=GroupKFold(n_splits=min(n_splits,len(set(groups))))
    for name,clf in [("logistic",LogisticRegression(class_weight="balanced",max_iter=400,random_state=seed)),("linear_svm",LinearSVC(class_weight="balanced",random_state=seed)),("complement_nb",ComplementNB())]:
        pipe=make_pipeline(TfidfVectorizer(ngram_range=(1,2)),clf)
        try: scores=cross_val_score(pipe,texts,labels,groups=groups,cv=cv,scoring="f1_macro")
        except ValueError: continue
        results[name]={"macro_f1_mean":float(scores.mean()),"macro_f1_std":float(scores.std())}
    return results

def compare_text_channels(labels, seed=7):
    """Grouped out-of-fold hierarchical comparison of user-intent/context weighting."""
    import numpy as np
    from sklearn.model_selection import GroupKFold
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.svm import LinearSVC
    from sklearn.metrics import f1_score
    if len(labels)<10: return {}
    groups=np.asarray([x.get("archetype_id",x["session_id"]) for x in labels])
    n_splits=min(5,len(set(groups)))
    if n_splits<2: return {}
    intent=[x.get("user_intent") or x.get("local_text","") for x in labels]
    context=[x.get("local_text","") for x in labels]
    modes={"intent_only":intent,"context_only":context,
           "intent_2x_plus_context":[f"{i} {i} {c}" for i,c in zip(intent,context)]}
    y_use=np.asarray([x["use_case_id"] for x in labels]); y_work=np.asarray([x["workflow_id"] for x in labels])
    results={}
    for mode,texts in modes.items():
        pred_use=[None]*len(labels); pred_work=[None]*len(labels)
        for tr,te in GroupKFold(n_splits=n_splits).split(texts,y_use,groups):
            vec=TfidfVectorizer(ngram_range=(1,2),min_df=1,sublinear_tf=True,max_features=50000)
            Xtr=vec.fit_transform([texts[i] for i in tr]); Xte=vec.transform([texts[i] for i in te])
            uc=LinearSVC(class_weight="balanced",random_state=seed).fit(Xtr,y_use[tr]); use_pred=uc.predict(Xte)
            wf_models={}
            for parent in set(y_use[tr]):
                loc=np.where(y_use[tr]==parent)[0]
                counts=Counter(y_work[tr][loc]); supported=[j for j in loc if counts[y_work[tr][j]]>=3]
                classes=set(y_work[tr][supported])
                if len(classes)==1: wf_models[parent]=next(iter(classes))
                elif len(classes)>1:
                    wf_models[parent]=LinearSVC(class_weight="balanced",random_state=seed).fit(Xtr[supported],y_work[tr][supported])
            for j,idx in enumerate(te):
                pred_use[idx]=use_pred[j]
                model=wf_models.get(use_pred[j])
                pred_work[idx]=(str(model.predict(Xte[j])[0]) if hasattr(model,"predict") else model) if model is not None else "needs_review"
        results[mode]={"use_case_macro_f1":float(f1_score(y_use,pred_use,average="macro",zero_division=0)),
                       "workflow_macro_f1":float(f1_score(y_work,pred_work,average="macro",zero_division=0)),
                       "workflow_candidate_coverage":sum(x!="needs_review" for x in pred_work)/len(pred_work),
                       "auto_accepted_coverage":0.0,
                       "status_policy":"All residual ML suggestions remain needs_review until calibrated and independently verified.",
                       "split":"5-fold GroupKFold by archetype; direct GLM training labels only"}
    best=max(results,key=lambda k:(results[k]["use_case_macro_f1"]+results[k]["workflow_macro_f1"],results[k]["use_case_macro_f1"]))
    return {"feature_views":results,"selected_view":best,"source_quality":"GLM pseudo-label agreement; not ground-truth accuracy"}
