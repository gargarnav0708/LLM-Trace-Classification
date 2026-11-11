"""Create a source-only V07 ZIP and verify integrity/size."""
from pathlib import Path
import shutil, zipfile

def create_zip(source_root,analysis_output,zip_path,max_bytes=5*1024*1024):
    source_root=Path(source_root); analysis_output=Path(analysis_output); zip_path=Path(zip_path)
    stage=zip_path.parent/".v07_stage"
    if stage.exists(): shutil.rmtree(stage)
    stage.mkdir(parents=True)
    (stage/"code").mkdir()
    ignore=shutil.ignore_patterns("__pycache__","*.pyc",".git","*.sqlite",".env")
    shutil.copytree(source_root/"v07",stage/"code"/"v07",ignore=ignore)
    shutil.copytree(source_root/"tests",stage/"code"/"tests",ignore=ignore)
    shutil.copy2(source_root/"v07"/"requirements.txt",stage/"requirements.txt")
    shutil.copy2(source_root/"README.md",stage/"README.md")
    report=source_root/"outputs"/"ASSESSMENT_REPORT.md"
    if report.exists(): shutil.copy2(report,stage/"ASSESSMENT_REPORT.md")
    shutil.copytree(analysis_output,stage/"output")
    if len((stage/"README.md").read_text().split())>300: raise ValueError("README exceeds 300 words")
    zip_path.parent.mkdir(parents=True,exist_ok=True)
    if zip_path.exists(): zip_path.unlink()
    with zipfile.ZipFile(zip_path,"w",zipfile.ZIP_DEFLATED,compresslevel=9) as z:
        for p in sorted(stage.rglob("*")):
            if p.is_file(): z.write(p,p.relative_to(stage).as_posix())
    with zipfile.ZipFile(zip_path) as z:
        bad=z.testzip()
        if bad: raise ValueError(f"Corrupt ZIP member: {bad}")
        names=z.namelist()
        forbidden=[n for n in names if any(x in n for x in ("traces.parquet",".sqlite",".env","__pycache__",".git/"))]
        if forbidden: raise ValueError(f"Forbidden package content: {forbidden[0]}")
    size=zip_path.stat().st_size
    if size>max_bytes: raise ValueError(f"ZIP is {size} bytes; limit is {max_bytes}")
    shutil.rmtree(stage)
    return {"path":str(zip_path),"size_bytes":size,"members":len(names),"integrity":"ok"}
