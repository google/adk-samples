# Copyright 2026 Google LLC. This software is provided as-is, without warranty or representation.
"""Wrapper script to deploy the Economic Research Agent with API keys loaded from .env."""

import os
import subprocess
from dotenv import dotenv_values

def deploy():
    # Load all variables from .env
    env_vars = dotenv_values(".env")
    
    # Select keys to deploy
    keys_to_deploy = [
        "BEA_API_KEY", "FRED_API_KEY", "CENSUS_API_KEY", 
        "BLS_API_KEY", "HUD_API_KEY", "FEC_API_KEY", 
        "EIA_API_KEY", "NEWS_API_KEY", "SERPER_API_KEY"
    ]
    
    deploy_env_list = []
    for k in keys_to_deploy:
        val = env_vars.get(k)
        if val:
            # Clean quotes if any
            val = val.strip().replace('"', '').replace("'", "")
            deploy_env_list.append(f"{k}={val}")
            
    env_str = ",".join(deploy_env_list)
    
    # Construct deploy command
    cmd = [
        "uv", "run", "agents-cli", "deploy",
        "--no-confirm-project",
        "--update-env-vars", env_str
    ]
    
    print("🚀 Starting deployment to Agent Runtime with local API keys...")
    print(f"Command: {' '.join(cmd)[:200]}... [truncated keys]")
    
    result = subprocess.run(cmd, capture_output=False)
    if result.returncode == 0:
        print("✅ Deployed and configured with environment variables successfully!")
    else:
        print(f"❌ Deployment failed with exit code: {result.returncode}")

if __name__ == "__main__":
    deploy()
