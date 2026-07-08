import os
import json
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
import threading
import google.auth
import google.auth.transport.requests

REMOTE_ENGINE_ID = "https://us-east1-aiplatform.googleapis.com/v1beta1/projects/697625214430/locations/us-east1/reasoningEngines/8517890192101605376"
CONCURRENCY = 3  # Reduced from 10 to avoid API rate limits

credentials, _ = google.auth.default()
token_lock = threading.Lock()

def get_active_token():
    with token_lock:
        # Check validity and refresh if expired or close to expiration
        if not credentials.valid:
            credentials.refresh(google.auth.transport.requests.Request())
        return credentials.token

def run_single_query(idx, q):
    token = get_active_token()
    cmd = [
        "uv", "run", "agents-cli", "run",
        "--url", REMOTE_ENGINE_ID,
        "--mode", "adk",
        "-H", f"Authorization: Bearer {token}",
        q
    ]
    
    max_retries = 3
    backoff = 3
    for attempt in range(max_retries + 1):
        start_time = time.time()
        try:
            res = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
            duration = time.time() - start_time
            
            stdout_content = res.stdout or ""
            stderr_content = res.stderr or ""
            
            # Check for 429 Rate Limit
            if "429" in stdout_content or "429" in stderr_content or "limit exceeded" in stdout_content.lower() or "limit exceeded" in stderr_content.lower():
                if attempt < max_retries:
                    sleep_time = (backoff ** attempt) + (idx % 4)  # Add jitter
                    print(f"⚠️ Query {idx} hit 429 rate limit. Retrying in {sleep_time}s... (Attempt {attempt+1}/{max_retries})")
                    time.sleep(sleep_time)
                    continue
            
            # Check for typical indicators of failure in output
            failure_keywords = [
                "couldn't fetch", "failed to fetch", "error", "exception",
                "unauthorized", "api key is not valid", "cannot import name",
                "not allowed", "limit exceeded", "timeout"
            ]
            
            is_failed = res.returncode != 0
            error_msg = ""
            
            if is_failed:
                error_msg = stderr_content or stdout_content
            else:
                for kw in failure_keywords:
                    if kw in stdout_content.lower() or kw in stderr_content.lower():
                        is_failed = True
                        error_msg = f"Potential data fetch failure found. Matched keyword: '{kw}'"
                        break
                        
            return {
                "index": idx,
                "query": q,
                "status": "FAILED" if is_failed else "SUCCESS",
                "duration_s": round(duration, 2),
                "response": stdout_content[:400] + "..." if len(stdout_content) > 400 else stdout_content,
                "error": error_msg
            }
        except subprocess.TimeoutExpired:
            if attempt < max_retries:
                time.sleep(5)
                continue
            return {
                "index": idx,
                "query": q,
                "status": "FAILED",
                "duration_s": 120.0,
                "error": "Query execution timed out after 120 seconds."
            }
        except Exception as e:
            return {
                "index": idx,
                "query": q,
                "status": "FAILED",
                "duration_s": round(time.time() - start_time, 2),
                "error": str(e)
            }


def run_bulk_simulations():
    evalset_path = "tests/eval/evalsets/wow_stress_test.evalset.json"
    if not os.path.exists(evalset_path):
        print(f"❌ Evalset file not found: {evalset_path}")
        return
        
    with open(evalset_path) as f:
        evalset = json.load(f)
        
    cases = evalset.get("eval_cases", [])
    queries = [case["conversation"][0]["user_content"]["parts"][0]["text"] for case in cases]
    
    print(f"Loaded {len(queries)} simulation queries from {evalset_path}")
    
    # Pre-verify token generation once at start
    print("🔑 Resolving Google Cloud OAuth Access Token...")
    try:
        get_active_token()
        print("🟢 Token authentication verified successfully.")
    except Exception as e:
        print(f"❌ Authentication failed: {e}")
        print("Please run `gcloud auth application-default login` to authenticate.")
        return
            
    print(f"🚀 Starting concurrent run with {CONCURRENCY} workers...\n")
    
    results = []
    failed_cases = []
    completed = 0
    
    start_all = time.time()
    
    with ThreadPoolExecutor(max_workers=CONCURRENCY) as executor:
        # Submit all tasks
        future_to_query = {executor.submit(run_single_query, i+1, q): q for i, q in enumerate(queries)}
        
        for future in as_completed(future_to_query):
            res_data = future.result()
            results.append(res_data)
            completed += 1
            
            if res_data["status"] == "FAILED":
                failed_cases.append(res_data)
                print(f"🔴 [{completed}/{len(queries)}] Query {res_data['index']} FAILED: {res_data['query'][:60]}... -> {res_data['error'][:100]}")
            else:
                if completed % 50 == 0 or completed == len(queries):
                    print(f"🟢 [{completed}/{len(queries)}] Completed. Elapsed time: {round(time.time() - start_all, 1)}s")
                    
    total_duration = time.time() - start_all
    print(f"\n🏁 Finished running {len(queries)} simulations in {round(total_duration/60, 2)} minutes.")
    print(f"✅ Success Rate: {len(queries) - len(failed_cases)} / {len(queries)}")
    
    # Write all results to file
    out_path = "tests/bulk_simulation_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
        
    # Write only failed/warn cases for reporting
    fail_report_path = "tests/bulk_simulation_failures.json"
    with open(fail_report_path, "w") as f:
        json.dump(failed_cases, f, indent=2)
        
    print(f"💾 Logs saved to:")
    print(f"   - Full Results: {out_path}")
    print(f"   - Failure Report: {fail_report_path}")

if __name__ == "__main__":
    run_bulk_simulations()
