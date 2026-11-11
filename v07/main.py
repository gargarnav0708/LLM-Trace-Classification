import argparse, json, os
from .pipeline import run

def main():
    p=argparse.ArgumentParser(description="Hyde V07 trace mining pipeline")
    p.add_argument("--input",required=True,help="Parquet, JSON, or JSONL call records")
    p.add_argument("--output",default="output")
    p.add_argument("--live-glm",action="store_true",help="Enable authorized GLM taxonomy discovery and annotation")
    p.add_argument("--budget",type=int,default=300)
    p.add_argument("--residual-budget",type=int,default=10000,help="Maximum residual sessions eligible for time-bounded GLM labeling")
    p.add_argument("--max-archetypes",type=int,default=30)
    p.add_argument("--overwrite",action="store_true",help="Replace generated output files in the chosen directory")
    p.add_argument("--taxonomy-file",help="Reuse a saved taxonomy JSON without discovery calls")
    p.add_argument("--reuse-annotations-from",help="Replay annotations_train.json and annotations_validation.json from this directory without GLM calls")
    args=p.parse_args()
    result=run(args.input,args.output,args.live_glm,args.budget,args.max_archetypes,args.overwrite,args.taxonomy_file,args.reuse_annotations_from,args.residual_budget)
    print(json.dumps(result,indent=2))

if __name__=="__main__": main()
