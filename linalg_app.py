#!/usr/bin/env python3
"""
LinAlg-Bench · Pipeline Manager

Run from repo root:
    uv run streamlit run linalg_app.py --server.port 8512

Folder structure enforced by this app:
    data/
      linalg_bench_3x3/4x4/5x5.csv   ← benchmark inputs
      output/
        {Model}/
          {Model}_results.jsonl        ← Stage 1 raw
          {Model}_*_failures.jsonl     ← Stage 1 failures  →  Stage 2 input
          {Model}_summary.csv
          inference.log
          judge/
            {subcat}_judge_labels.csv     ← Stage 2 output  →  Stage 3 input
            {subcat}_judge_validated.csv  ← Stage 3 output
      results/
        accuracy_3x3.csv   ← summarize output
        accuracy_4x4.csv
        accuracy_5x5.csv
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path


# ─────────────────────────────────────────────────────────────────────
import pandas as pd
import streamlit as st
import time
import threading
from concurrent.futures import ThreadPoolExecutor

# CONFIG
# ─────────────────────────────────────────────────────────────────────

BASE_DIR     = Path(__file__).parent
PIPELINE_DIR = BASE_DIR / "pipeline"
DATA_DIR     = BASE_DIR / "data"
OUTPUT_DIR   = DATA_DIR / "output"
RESULTS_DIR  = DATA_DIR / "results"

try:
    from dotenv import load_dotenv
    load_dotenv(BASE_DIR / ".env")
except ImportError:
    pass

_API_KEY = os.environ.get("GEMINI_API_KEY", "")

JUDGE_MODEL_OPTIONS = [
    "gemini-3.1-flash-lite-preview",
    "gemini-3.1-pro-preview",
]

INFERENCE_MODELS = [
    "DeepSeek-V3", "GPT-4o", "GPT-5.2", "Mistral-Large",
    "Qwen2.5-72B", "Qwen3-235B", "Llama-3.3-70B",
    "Claude-4.5-Sonnet", "OpenAI-o1", "Gemini-3.0-Pro",
]

REQUIRED_COLS = ["problem_latex", "answer_latex"]
ID_ALIASES    = ["id", "problem_id", "question_id", "Problem_ID"]
OPTIONAL_COLS = ["problem_text", "subcategory", "subcat"]

_VALID_SUBCATS = {"det", "eig", "rank", "null", "inv", "mult", "pow", "vec", "trans", "trace"}
_SUBCAT_MAP = {
    "determinant": "det",   "det": "det",
    "eigenvalue": "eig",    "eigenvalues": "eig", "eig": "eig", "eigen": "eig",
    "rank": "rank",
    "nullity": "null",      "null": "null",
    "inverse": "inv",       "inv": "inv",
    "multiplication": "mult","mult": "mult",
    "matrix_power": "pow",  "pow": "pow",
    "matrix_vector": "vec", "vec": "vec", "vector_product": "vec",
    "transpose": "trans",   "trans": "trans",
    "trace": "trace",
}

STATUS_ICON = {
    "idle": "⚪", "starting": "🟡", "running": "🔵",
    "done": "✅", "failed": "❌", "killed": "⛔",
}

# ─────────────────────────────────────────────────────────────────────
# SESSION STATE
# ─────────────────────────────────────────────────────────────────────

def init_session():
    if "jobs" not in st.session_state:
        st.session_state.jobs = {}
    if "refresh_counter" not in st.session_state:
        st.session_state.refresh_counter = 0

init_session()

# ─────────────────────────────────────────────────────────────────────
# JOB MONITOR FRAGMENT - auto-refreshes every 2 seconds when job running
# ─────────────────────────────────────────────────────────────────────

@st.fragment(run_every=2)
def job_monitor():
    infer_jobs = {k: v for k, v in st.session_state.jobs.items() if k.startswith("S1__")}
    if infer_jobs:
        st.markdown("**📊 Progress:**")
        for jk, jv in infer_jobs.items():
            istatus = jv.get("status", "idle")
            is_running = istatus in ("running", "starting")
            elapsed = (jv.get("end_time") or time.time()) - jv.get("start_time", time.time())
            label = jk.replace("S1__", "").replace("__", "  ·  ")
            icon = "🔵" if is_running else ("✅" if istatus == "done" else "⚪")
            st.markdown(f"**{icon} {label}** — {istatus.upper()} — {elapsed:.0f}s")
            log_lines = jv.get("log", [])
            if log_lines:
                st.code("\n".join(log_lines[-20:]), language="log")

@st.fragment(run_every=2)
def job_monitor_s2():
    s2_jobs = {k: v for k, v in st.session_state.jobs.items() if k.startswith("S2__")}
    if s2_jobs:
        st.markdown("**📊 Progress:**")
        for jk, jv in s2_jobs.items():
            istatus = jv.get("status", "idle")
            is_running = istatus in ("running", "starting")
            elapsed = (jv.get("end_time") or time.time()) - jv.get("start_time", time.time())
            label = jk.replace("S2__", "").replace("__", "  ·  ")
            icon = "🔵" if is_running else ("✅" if istatus == "done" else "⚪")
            st.markdown(f"**{icon} {label}** — {istatus.upper()} — {elapsed:.0f}s")
            log_lines = jv.get("log", [])
            if log_lines:
                st.code("\n".join(log_lines[-20:]), language="log")

@st.fragment(run_every=2)
def job_monitor_s3():
    s3_jobs = {k: v for k, v in st.session_state.jobs.items() if k.startswith("S3__")}
    if s3_jobs:
        st.markdown("**📊 Progress:**")
        for jk, jv in s3_jobs.items():
            istatus = jv.get("status", "idle")
            is_running = istatus in ("running", "starting")
            elapsed = (jv.get("end_time") or time.time()) - jv.get("start_time", time.time())
            label = jk.replace("S3__", "").replace("__", "  ·  ")
            icon = "🔵" if is_running else ("✅" if istatus == "done" else "⚪")
            st.markdown(f"**{icon} {label}** — {istatus.upper()} — {elapsed:.0f}s")
            log_lines = jv.get("log", [])
            if log_lines:
                st.code("\n".join(log_lines[-20:]), language="log")

# ─────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────

def _rel(p: Path) -> str:
    try:    return str(p.relative_to(BASE_DIR))
    except ValueError: return str(p)

def _has_format_in_path(p: Path, base: Path) -> bool:
    return any("format" in part.lower() or "sensitivity" in part.lower() or "model" in part.lower() for part in p.relative_to(base).parts)

def count_records(path: Path) -> int:
    if not path.exists():
        return 0
    with open(path, encoding="utf-8") as f:
        return sum(1 for line in f if line.strip())

def get_subcat_from_jsonl(path: Path) -> str:
    try:
        with open(path, encoding="utf-8") as f:
            first = json.loads(f.readline())
        for id_field in ("Problem_ID", "question_id"):
            id_val = first.get(id_field, "")
            if id_val:
                parts = str(id_val).split("_")
                if len(parts) >= 3 and parts[2] in _VALID_SUBCATS:
                    return parts[2]
        subcat_raw = first.get("Subcat") or first.get("subcat") or ""
        if subcat_raw:
            return _SUBCAT_MAP.get(subcat_raw.lower(), subcat_raw)
    except Exception:
        return ""
    return ""

def csv_stats(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        df = pd.read_csv(path, on_bad_lines="skip")
        stats: dict = {"rows": len(df)}
        if "Error_Tag" in df.columns:
            stats["tags"] = df["Error_Tag"].value_counts().to_dict()
        if "validated" in df.columns:
            stats["validation"] = df["validated"].value_counts().to_dict()
        return stats
    except Exception:
        return {"rows": 0}

def scan_input_files() -> list[Path]:
    if not DATA_DIR.exists():
        return []
    files = []
    for ext in ("*.csv", "*.xlsx", "*.xls"):
        files.extend(
            f for f in DATA_DIR.rglob(ext)
            if not _has_format_in_path(f, DATA_DIR)
            and "output"  not in f.relative_to(DATA_DIR).parts
            and "results" not in f.relative_to(DATA_DIR).parts
        )
    return sorted(set(files))

def scan_models() -> list[str]:
    if not OUTPUT_DIR.exists():
        return []
    return sorted(d.name for d in OUTPUT_DIR.iterdir() if d.is_dir())

def scan_failures_jsonl(model: str) -> list[Path]:
    model_dir = OUTPUT_DIR / model
    if not model_dir.exists():
        return []
    return sorted(model_dir.glob("*_failures.jsonl"))

def job_status(key: str) -> str:
    return st.session_state.jobs.get(key, {}).get("status", "idle")

# ─────────────────────────────────────────────────────────────────────
# JOB EXECUTION
# ─────────────────────────────────────────────────────────────────────

def run_job(job_dict: dict, cmd: list[str]):
    job_dict["status"]     = "running"
    job_dict["log"]        = []
    job_dict["start_time"] = time.time()
    env = {**os.environ, "PYTHONUNBUFFERED": "1"}
    try:
        proc = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, bufsize=1, cwd=str(BASE_DIR), env=env,
        )
        job_dict["proc"] = proc
        job_dict["_stop_event"] = threading.Event()
        
        while proc.poll() is None:
            line = proc.stdout.readline()
            if not line:
                if job_dict["_stop_event"].is_set():
                    break
                continue
            line = line.rstrip()
            job_dict["log"].append(line)
            job_dict["log_count"] = len(job_dict["log"])
        
        job_dict["returncode"] = proc.wait()
        job_dict["status"] = "done" if job_dict["returncode"] == 0 else "failed"
    except Exception as e:
        job_dict["log"].append(f"ERROR: {e}")
        job_dict["status"] = "failed"
    finally:
        job_dict["end_time"] = time.time()

def start_job(key: str, cmd: list[str]):
    job_dict = {
        "status": "starting", "proc": None,
        "cmd": " ".join(str(c) for c in cmd),
        "log": [], "start_time": time.time(),
        "end_time": None, "returncode": None,
    }
    st.session_state.jobs[key] = job_dict
    threading.Thread(target=run_job, args=(job_dict, cmd), daemon=True).start()

def stop_job(key: str):
    job  = st.session_state.jobs.get(key)
    proc = job.get("proc") if job else None
    if proc:
        try:
            proc.kill()
            job["status"] = "killed"
            job["log"].append("[KILLED] Process terminated")
        except Exception as e:
            job["log"].append(f"[ERROR] Could not kill: {e}")

def render_job_log(key: str, max_lines: int = 40):
    job = st.session_state.jobs.get(key, {})
    if not job:
        return
    log_lines = job.get("log", [])
    st.code("\n".join(log_lines[-max_lines:]) if log_lines else "Waiting for output...",
            language="log")

# ─────────────────────────────────────────────────────────────────────
# PAGE CONFIG + CSS
# ─────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="LinAlg-Bench Pipeline",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
[data-testid="stSidebar"] { min-width: 350px; max-width: 420px; }
div[data-baseweb="popover"] ul { max-height: 500px !important; }
div[data-baseweb="select"] > div { white-space: normal !important; word-break: break-all; }

.file-ok   { color: #2e7d32; font-weight: 600; }
.file-miss { color: #b71c1c; font-weight: 600; }

div[data-testid="stMetricValue"] { font-size: 1.4rem !important; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## ⚙️ Settings")
    st.divider()

    with st.expander("🔑 API Keys", expanded=False):
        gemini_key     = st.text_input("GEMINI_API_KEY",     value=_API_KEY,
                                        type="password", key="sb_gemini")
        openrouter_key = st.text_input("OPENROUTER_API_KEY",
                                        value=os.environ.get("OPENROUTER_API_KEY", ""),
                                        type="password", key="sb_or")
        openai_key     = st.text_input("OPENAI_API_KEY",
                                        value=os.environ.get("OPENAI_API_KEY", ""),
                                        type="password", key="sb_oai")
        if gemini_key:     os.environ["GEMINI_API_KEY"]     = gemini_key
        if openrouter_key: os.environ["OPENROUTER_API_KEY"] = openrouter_key
        if openai_key:     os.environ["OPENAI_API_KEY"]     = openai_key

    st.divider()
    st.markdown("**🔧 Stage 1 — Inference**")
    
    input_files = scan_input_files()
    if not input_files:
        st.warning("No CSV/Excel files found in `data/`.")
    else:
        input_opts = {_rel(f): f for f in input_files}
        infer_filter = st.text_input("🔎 Filter input", placeholder="e.g. 4x4", key="sb_s1_filter")
        if infer_filter:
            input_opts = {k: v for k, v in input_opts.items()
                        if infer_filter.lower() in k.lower()}
        if not input_opts:
            st.warning("No files match filter.")
        else:
            sb_src_path = st.selectbox("📥 Input file", list(input_opts.keys()), key="sb_s1_input")
            src_path = input_opts[sb_src_path]
            
            # Validation
            src_path_valid = False
            if src_path and src_path.exists():
                try:
                    df_p = (pd.read_excel(src_path, nrows=2)
                            if src_path.suffix.lower() in (".xlsx", ".xls")
                            else pd.read_csv(src_path, nrows=2))
                    cols_norm = [c.strip().lower() for c in df_p.columns]
                    has_id = any(a.lower() in cols_norm for a in ID_ALIASES)
                    has_req = all(c.lower() in cols_norm for c in REQUIRED_COLS)
                    if has_id and has_req:
                        st.caption(f"✅ {len(df_p.columns)} columns")
                        src_path_valid = True
                except:
                    pass
            
            if src_path_valid:
                infer_model = st.selectbox("🤖 Model", INFERENCE_MODELS, index=0, key="sb_s1_model")
                infer_limit = st.number_input("📏 Limit", min_value=0, value=5, step=1, key="sb_s1_limit")
                infer_dry_run = st.checkbox("🧪 Dry Run", value=False, key="sb_s1_dryrun")

    st.divider()
    with st.expander("🤖 Judge Models", expanded=False):
        judge_model_s2 = st.selectbox("Stage 2 model", JUDGE_MODEL_OPTIONS, index=0, key="sb_jm2")
        judge_model_s3 = st.selectbox("Stage 3 model", JUDGE_MODEL_OPTIONS,
                                       index=1 if len(JUDGE_MODEL_OPTIONS) > 1 else 0, key="sb_jm3")
        s2_dry_run = st.checkbox("S2 Dry Run", value=False, key="sb_s2_dry")
        s3_dry_run = st.checkbox("S3 Dry Run", value=False, key="sb_s3_dry")
        limit_val = st.number_input("Limit (0=all)", min_value=0, value=0, step=1, key="sb_limit")
        limit = int(limit_val) if limit_val > 0 else None
        resume = st.checkbox("Resume", value=True, key="sb_resume")

    st.divider()
    st.markdown("**🧪 API Key Check**")
    if st.button("Validate Gemini Key", use_container_width=True, key="sb_validate"):
        if str(PIPELINE_DIR) not in sys.path:
            sys.path.insert(0, str(PIPELINE_DIR))
        try:
            from judge_llm import define_clients, call_llm
            for label, mid in [("S2", judge_model_s2), ("S3", judge_model_s3)]:
                with st.spinner(f"Testing {mid}..."):
                    try:
                        client = define_clients(mid)
                        raw, _ = call_llm("You are helpful.",
                                          "Capital of France? One word.", mid, client)
                        if "paris" in raw.lower():
                            st.success(f"✅ {label}: valid")
                        else:
                            st.warning(f"⚠️ {label}: {raw.strip()[:60]}")
                    except Exception as e:
                        st.error(f"❌ {label}: {e}")
        except ImportError as e:
            st.error(f"Import error: {e}")

# ─────────────────────────────────────────────────────────────────────
# HEADER + PIPELINE STATUS STRIP
# ─────────────────────────────────────────────────────────────────────

st.markdown("# 🔬 LinAlg-Bench Pipeline Manager")

# Quick status strip — count completed jobs per stage
s1_done = sum(1 for k, v in st.session_state.jobs.items()
              if k.startswith("S1__") and v.get("status") == "done")
s2_done = sum(1 for k, v in st.session_state.jobs.items()
              if k.startswith("S2__") and v.get("status") == "done")
s3_done = sum(1 for k, v in st.session_state.jobs.items()
              if k.startswith("S3__") and v.get("status") == "done")
acc_files_count = len(list(RESULTS_DIR.glob("accuracy_*.csv"))) if RESULTS_DIR.exists() else 0

strip = st.columns(4)
strip[0].metric("Stage 1 · Inference",   f"{s1_done} run{'s' if s1_done != 1 else ''}")
strip[1].metric("Stage 2 · Build Judge", f"{s2_done} run{'s' if s2_done != 1 else ''}")
strip[2].metric("Stage 3 · Validate",    f"{s3_done} run{'s' if s3_done != 1 else ''}")
strip[3].metric("Accuracy files",        str(acc_files_count))

st.divider()

# ─────────────────────────────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────────────────────────────

tab1, tab2, tab3, tab_results, tab_logs = st.tabs([
    "🚀 Stage 1 — Inference",
    "🔍 Stage 2 — Build Judge",
    "✅ Stage 3 — Validate",
    "📊 Results",
    "📝 Logs",
])

# ══════════════════════════════════════════════════════════════════════
# TAB 1 — INFERENCE
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    st.subheader("🚀 Stage 1 — Inference")
    st.caption(f"Saves to → `data/output/{{Model}}/`")
    
    # Run button at top - small
    if st.button("▶ Run", key="t1_run", disabled=not src_path_valid):
        model_tag = f"{infer_model}_dryrun" if infer_dry_run else infer_model
        infer_out = OUTPUT_DIR / model_tag
        cmd = [
            "uv", "run", "python", str(PIPELINE_DIR / "inference.py"),
            "--input",  str(src_path),
            "--model",  infer_model,
            "--output", str(infer_out),
        ]
        if infer_dry_run:   cmd.append("--dry-run")
        if infer_limit > 0: cmd.extend(["--limit", str(infer_limit)])
        jkey = f"S1__{infer_model}__{src_path.stem}"
        start_job(jkey, cmd)
        st.rerun()
    
    # Job monitor below Run button - shows progress
    job_monitor()
    
    st.divider()
    
    # Use sidebar selections - display only what was set
    st.markdown("**📥 Input:** " + (src_path.name if src_path else "—"))
    
    if sb_src_path and src_path_valid:
        st.markdown(f"**🤖 Model:** {infer_model}")
        if infer_dry_run:
            st.caption("🧪 Dry Run")
        if infer_limit:
            st.caption(f"📏 Limit: {infer_limit}")

    # Output folder browser
    models_found = scan_models()
    if models_found:
        st.divider()
        st.markdown("**📁 Output Folder**")
        for m in models_found:
            mdir  = OUTPUT_DIR / m
            files = sorted(f for f in mdir.iterdir() if f.is_file())
            subdirs = sorted(d for d in mdir.iterdir() if d.is_dir())
            with st.expander(f"📂 `output/{m}/`  —  {len(files)} files"):
                for f in files:
                    st.caption(f"📄 {f.name}  ({f.stat().st_size:,} B)")
                for d in subdirs:
                    sub_files = list(d.glob("*"))
                    st.caption(f"📁 {d.name}/  ({len(sub_files)} files)")

# ══════════════════════════════════════════════════════════════════════
# TAB 2 — BUILD JUDGE
# ══════════════════════════════════════════════════════════════════════
with tab2:
    st.subheader("🔍 Stage 2 — Build Judge")
    st.caption("Input: failures JSONL  ·  Output: `data/output/{Model}/judge/{subcat}_judge_labels.csv`")
    
    models_s2 = scan_models()
    if not models_s2:
        st.warning("No output found in `data/output/`. Run Stage 1 first.")
        st.stop()

    t2_left, t2_right = st.columns([1, 1])

    with t2_left:
        sel_model_s2  = st.selectbox("📂 Model", models_s2, key="t2_model")
        failures_list = scan_failures_jsonl(sel_model_s2)

        if not failures_list:
            # Check if Stage 1 ran at all and why there are no failures
            results_jsonl = OUTPUT_DIR / sel_model_s2 / f"{sel_model_s2}_results.jsonl"
            summary_csv   = OUTPUT_DIR / sel_model_s2 / f"{sel_model_s2}_summary.csv"
            if results_jsonl.exists():
                # Stage 1 ran — check if model got everything correct
                try:
                    df_s = pd.read_csv(summary_csv) if summary_csv.exists() else None
                    if df_s is not None and "accuracy" in df_s.columns.str.lower().tolist():
                        acc_col = [c for c in df_s.columns if c.lower() == "accuracy"][0]
                        acc_val = df_s[acc_col].mean()
                        if acc_val >= 99.0:
                            st.success(f"🎉 No failures — model scored ~{acc_val:.1f}% "
                                       "on this run. Nothing to judge!")
                        else:
                            st.warning(f"Stage 1 results exist but no failures JSONL was written. "
                                       f"Accuracy was {acc_val:.1f}% — re-run Stage 1 with a "
                                       "larger dataset to generate failures.")
                    else:
                        # Count failures manually from results JSONL
                        n_fail = sum(
                            1 for line in open(results_jsonl, encoding="utf-8")
                            if line.strip() and not json.loads(line).get("correct", True)
                        )
                        if n_fail == 0:
                            st.success("🎉 No failures found in results — model answered all "
                                       "records correctly on this run.")
                        else:
                            st.warning(f"Found {n_fail} failures in results JSONL but no "
                                       "failures JSONL written. Re-run Stage 1.")
                except Exception:
                    st.warning("Stage 1 results exist but no failures file found. "
                               "Try re-running Stage 1 with more records.")
            else:
                st.warning(f"No Stage 1 output in `output/{sel_model_s2}/`. Run Stage 1 first.")
            sel_failures = None
            subcat_s2    = None
        else:
            fail_opts    = {_rel(f): f for f in failures_list}
            sel_fail_key = st.selectbox("📄 Failures JSONL", list(fail_opts.keys()), key="t2_jsonl")
            sel_failures = fail_opts[sel_fail_key]
            subcat_s2    = get_subcat_from_jsonl(sel_failures)

    with t2_right:
        if sel_failures and sel_failures.exists():
            n_fail = count_records(sel_failures)
            st.metric("Failures", n_fail)
            if subcat_s2 and subcat_s2 in _VALID_SUBCATS:
                st.success(f"Subcat: **{subcat_s2}**")
            elif subcat_s2:
                st.warning(f"Subcat: `{subcat_s2}`")
                st.caption("Multi-subcat JSONL — judge will use first detected subcat")
            else:
                st.error("Could not detect subcat")
                subcat_s2 = None

    if sel_failures and sel_failures.exists() and subcat_s2:
        judge_out_dir    = OUTPUT_DIR / sel_model_s2 / "judge"
        judge_out_dir.mkdir(parents=True, exist_ok=True)
        judge_labels_csv = judge_out_dir / f"{subcat_s2}_judge_labels.csv"

        st.divider()
        io_c1, io_c2 = st.columns([1, 1])
        with io_c1:
            st.markdown("**Input / Output**")
            st.caption(f"📥 Input:  `{_rel(sel_failures)}`  ({count_records(sel_failures)} records)")
            if judge_labels_csv.exists():
                stats = csv_stats(judge_labels_csv)
                st.success(f"✅ Output exists: `{_rel(judge_labels_csv)}` "
                           f"({stats.get('rows','?')} rows)")
                if "tags" in stats:
                    st.json(stats["tags"], expanded=False)
            else:
                st.info(f"⬜ Output will be: `{_rel(judge_labels_csv)}`")

        with io_c2:
            s2_key    = f"S2__{sel_model_s2}__{subcat_s2}" if subcat_s2 else ""
            s2_status = job_status(s2_key) if s2_key else "idle"
            st.markdown("**Status**")
            st.write(f"{STATUS_ICON.get(s2_status,'⚪')} {s2_status.upper()}")
            if s2_status in ("running", "starting"):
                if st.button("⏹ Stop", key=f"t2_stop_{s2_key}"):
                    stop_job(s2_key); st.rerun()
                job_monitor_s2()  # progress in right pane
            elif s2_status == "idle" and subcat_s2:
                if st.button("▶ Run Stage 2", use_container_width=True, key="t2_run"):
                    cmd = [
                        "uv", "run", "python", str(PIPELINE_DIR / "build_judge.py"),
                        "--results",      str(sel_failures),
                        "--output",       str(judge_labels_csv),
                        "--subcat",       subcat_s2,
                        "--judge-llm-id", judge_model_s2,
                    ]
                    if s2_dry_run: cmd.append("--dry-run")
                    if limit:   cmd.extend(["--limit", str(limit)])
                    if resume:  cmd.append("--resume")
                    start_job(s2_key, cmd)
                    st.rerun()

# ══════════════════════════════════════════════════════════════════════
# TAB 3 — VALIDATE JUDGE
# ══════════════════════════════════════════════════════════════════════
with tab3:
    st.subheader("✅ Stage 3 — Validate Judge")
    st.caption("Input: Stage 2 CSV + failures JSONL  ·  "
               "Output: `data/output/{Model}/judge/{subcat}_judge_validated.csv`")
     
    models_s3 = scan_models()
    if not models_s3:
        st.warning("No output found. Run Stage 1 first.")
        st.stop()

    sel_model_s3 = st.selectbox("📂 Model", models_s3, key="t3_model")
    judge_dir_s3 = OUTPUT_DIR / sel_model_s3 / "judge"

    s2_csvs = sorted(judge_dir_s3.glob("*_judge_labels.csv")) if judge_dir_s3.exists() else []
    if not s2_csvs:
        st.warning(f"No Stage 2 CSV in `output/{sel_model_s3}/judge/`. Run Stage 2 first.")
        st.stop()

    t3_left, t3_right = st.columns([1, 1])

    with t3_left:
        s2_opts     = {_rel(f): f for f in s2_csvs}
        sel_s2_key  = st.selectbox("📄 Stage 2 CSV", list(s2_opts.keys()), key="t3_s2csv")
        sel_s2_csv  = s2_opts[sel_s2_key]
        subcat_s3   = sel_s2_csv.stem.replace("_judge_labels", "")

        # Find matching failures JSONL
        model_dir_s3 = OUTPUT_DIR / sel_model_s3
        fail_candidates = sorted(model_dir_s3.glob(f"*{subcat_s3}*failures*.jsonl"))
        if not fail_candidates:
            fail_candidates = sorted(model_dir_s3.glob("*_failures.jsonl"))

        if fail_candidates:
            fail_opts_s3  = {_rel(f): f for f in fail_candidates}
            sel_fail_s3_k = st.selectbox("📄 Failures JSONL", list(fail_opts_s3.keys()), key="t3_jsonl")
            fail_jsonl_s3 = fail_opts_s3[sel_fail_s3_k]
        else:
            st.error(f"No failures JSONL found in `output/{sel_model_s3}/`")
            fail_jsonl_s3 = None

    with t3_right:
        if sel_s2_csv.exists():
            stats = csv_stats(sel_s2_csv)
            st.metric("S2 rows", stats.get("rows", "?"))
        if fail_jsonl_s3 and fail_jsonl_s3.exists():
            st.metric("Failures", count_records(fail_jsonl_s3))

    if sel_s2_csv.exists() and fail_jsonl_s3 and fail_jsonl_s3.exists():
        validated_csv = judge_dir_s3 / f"{subcat_s3}_judge_validated.csv"

        st.divider()
        io_c1, io_c2 = st.columns([1, 1])
        with io_c1:
            st.markdown("**Input / Output**")
            st.caption(f"📥 Stage 2 CSV: `{_rel(sel_s2_csv)}`")
            st.caption(f"📥 Failures:    `{_rel(fail_jsonl_s3)}`")
            if validated_csv.exists():
                stats = csv_stats(validated_csv)
                st.success(f"✅ Output exists: `{_rel(validated_csv)}` "
                           f"({stats.get('rows','?')} rows)")
                if "validation" in stats:
                    st.json(stats["validation"], expanded=False)
            else:
                st.info(f"⬜ Output will be: `{_rel(validated_csv)}`")

        with io_c2:
            s3_key    = f"S3__{sel_model_s3}__{subcat_s3}" if subcat_s3 else ""
            s3_status = job_status(s3_key) if s3_key else "idle"
            st.markdown("**Status**")
            st.write(f"{STATUS_ICON.get(s3_status,'⚪')} {s3_status.upper()}")
            if s3_status in ("running", "starting"):
                if st.button("⏹ Stop", key=f"t3_stop_{s3_key}"):
                    stop_job(s3_key); st.rerun()
                job_monitor_s3()  # progress in right pane
            elif s3_status == "idle" and subcat_s3:
                if st.button("▶ Run Stage 3", use_container_width=True, key="t3_run"):
                    cmd = [
                        "uv", "run", "python", str(PIPELINE_DIR / "validate_judge.py"),
                        "--judge",        str(sel_s2_csv),
                        "--results",      str(fail_jsonl_s3),
                        "--output",       str(validated_csv),
                        "--subcat",       subcat_s3,
                        "--judge-llm-id", judge_model_s3,
                    ]
                    if s3_dry_run: cmd.append("--dry-run")
                    if limit:   cmd.extend(["--limit", str(limit)])
                    if resume:  cmd.append("--resume")
                    start_job(s3_key, cmd)
                    job_monitor_s3()  # show progress below button
                    st.rerun()

# ══════════════════════════════════════════════════════════════════════
# TAB 4 — RESULTS
# ══════════════════════════════════════════════════════════════════════
with tab_results:
    st.subheader("📊 Results")

    col_acc, col_val = st.columns(2)

    with col_acc:
        st.markdown("**Model Accuracy** (`data/results/`)")
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        acc_files = sorted(RESULTS_DIR.glob("accuracy_*.csv"))
        if not acc_files:
            st.info("No accuracy files yet. Run inference to generate them.")
        else:
            for acc_f in acc_files:
                try:
                    df_acc = pd.read_csv(acc_f)
                    st.markdown(f"`{acc_f.name}`")
                    st.dataframe(df_acc, use_container_width=True, hide_index=True)
                except Exception as e:
                    st.error(f"Cannot read {acc_f.name}: {e}")

    with col_val:
        st.markdown("**Stage 3 Validated Outputs**")
        validated_files = sorted(OUTPUT_DIR.rglob("*_judge_validated.csv")) \
            if OUTPUT_DIR.exists() else []
        if not validated_files:
            st.info("No validated outputs yet. Run Stage 3.")
        else:
            for vf in validated_files:
                stats = csv_stats(vf)
                st.markdown(f"`{_rel(vf)}`  — {stats.get('rows','?')} rows")
                if "validation" in stats:
                    st.json(stats["validation"], expanded=False)
                if "tags" in stats:
                    st.json(stats["tags"], expanded=False)

# ══════════════════════════════════════════════════════════════════════
# TAB 5 — LOGS
# ══════════════════════════════════════════════════════════════════════
with tab_logs:
    st.subheader("📝 Live Logs")

    lc1, lc2 = st.columns([3, 1])
    with lc2:
        if st.button("🗑 Clear finished", use_container_width=True, key="clear_jobs"):
            st.session_state.jobs = {
                k: v for k, v in st.session_state.jobs.items()
                if v.get("status") in ("running", "starting")
            }
            st.rerun()

    active_keys = [k for k, v in st.session_state.jobs.items() if v.get("log") is not None]
    if not active_keys:
        st.info("No jobs yet.")
    else:
        log_tabs = st.tabs([
            f"{STATUS_ICON.get(st.session_state.jobs[k].get('status','idle'),'⚪')} {k}"
            for k in active_keys
        ])
        for log_tab, jk in zip(log_tabs, active_keys):
            jv = st.session_state.jobs[jk]
            with log_tab:
                if jv.get("cmd"):
                    with st.expander("Command", expanded=False):
                        st.code(jv["cmd"], language="bash")
                s = jv.get("status", "idle")
                hc1, hc2, hc3 = st.columns([2, 1, 1])
                with hc1:
                    state = "running" if s == "running" else "complete" if s == "done" else "error"
                    with st.status(f"**{s.upper()}**",
                                   expanded=(s in ("running", "starting")), state=state):
                        if jv.get("start_time"):
                            elapsed = (jv.get("end_time") or time.time()) - jv["start_time"]
                            st.caption(f"Elapsed: {elapsed:.0f}s")

                with hc2:
                    if jv.get("returncode") is not None:
                        rc = jv["returncode"]
                        st.metric("Return code", rc, delta=None)
                with hc3:
                    if s in ("running", "starting"):
                        if st.button("⏹ Stop", key=f"log_stop_{jk}"):
                            stop_job(jk); st.rerun()
                st.code("\n".join(jv.get("log", [])), language="log")

# Auto-refresh - click Refresh button above to see progress

