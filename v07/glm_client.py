"""OpenAI-compatible authorized endpoint client with strict JSON and SQLite cache."""
import hashlib, json, os, sqlite3, time, urllib.request, urllib.error, logging
log=logging.getLogger("v07.glm")

class IncompleteResponse(ValueError):
    """The endpoint marked a generated response incomplete (for example, token limit)."""

class GLMClient:
    def __init__(self, api_key, base_url="https://inference.hyde.tech/v1", model="hyde/glm-5.3-flash",
                 cache_path="work/glm_cache.sqlite", timeout=90):
        self.api_key,self.base_url,self.model,self.timeout=api_key,base_url.rstrip("/"),model,timeout
        os.makedirs(os.path.dirname(cache_path) or ".",exist_ok=True)
        self.db=sqlite3.connect(cache_path)
        self.db.execute("CREATE TABLE IF NOT EXISTS responses (key TEXT PRIMARY KEY, response TEXT NOT NULL, created REAL NOT NULL)")
        self.db.commit()
    def _cached(self,key):
        row=self.db.execute("SELECT response FROM responses WHERE key=?",(key,)).fetchone()
        return json.loads(row[0]) if row else None
    def _save(self,key,obj):
        self.db.execute("INSERT OR REPLACE INTO responses VALUES(?,?,?)",(key,json.dumps(obj),time.time())); self.db.commit()
    @staticmethod
    def parse_json(content):
        if not content: raise ValueError("empty response content")
        try: return json.loads(content)
        except ValueError:
            start=content.find("{"); end=content.rfind("}")
            if start<0 or end<=start: raise ValueError("response is not valid JSON")
            return json.loads(content[start:end+1])
    def complete(self,system,user,max_tokens=1800):
        key=hashlib.sha256((self.base_url+self.model+system+user).encode()).hexdigest()
        cached=self._cached(key)
        if cached is not None: return cached
        if not self.api_key: raise RuntimeError("Set HYDE_API_KEY before live GLM requests")
        payload={"model":self.model,"temperature":0,"max_tokens":max_tokens,
                 "messages":[{"role":"system","content":system},{"role":"user","content":user}],
                 "response_format":{"type":"json_object"}}
        started=time.monotonic()
        try:
            raw=self._request(payload)
        except urllib.error.HTTPError as e:
            # Some OpenAI-compatible servers do not implement JSON mode. Retry once
            # without response_format only for a rejected request, never for timeouts.
            if e.code!=400 or "response_format" not in payload: raise
            payload.pop("response_format")
            raw=self._request(payload)
        choice=(raw.get("choices") or [{}])[0]; finish=choice.get("finish_reason")
        if finish=="length": raise IncompleteResponse("finish_reason=length; incomplete output")
        content=(choice.get("message") or {}).get("content")
        parsed=self.parse_json(content)
        parsed["_usage"]={"latency_seconds":round(time.monotonic()-started,3),"finish_reason":finish,
                           "prompt_tokens":(raw.get("usage") or {}).get("prompt_tokens"),
                           "completion_tokens":(raw.get("usage") or {}).get("completion_tokens")}
        self._save(key,parsed)
        log.info("GLM request ok latency=%.3fs prompt_tokens=%s completion_tokens=%s finish=%s",
                 parsed["_usage"]["latency_seconds"],parsed["_usage"]["prompt_tokens"],parsed["_usage"]["completion_tokens"],finish)
        return parsed
    def _request(self,payload):
        req=urllib.request.Request(self.base_url+"/chat/completions",data=json.dumps(payload).encode(),
             headers={"Authorization":"Bearer "+self.api_key,"Content-Type":"application/json"},method="POST")
        with urllib.request.urlopen(req,timeout=self.timeout) as res: return json.loads(res.read())
