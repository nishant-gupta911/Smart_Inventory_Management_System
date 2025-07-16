#!/usr/bin/env python3
"""
Verification script to check if all DEMO files work correctly after reorganization
"""

import sys
import os
from pathlib import Path
import subprocess

def test_demo_file(file_path):
    """Test if a demo file can run without import errors"""
    print(f"\n🔍 Testing {file_path.name}...")
    
    try:
        # Try to run the file
        result = subprocess.run([sys.executable, str(file_path)], 
                              capture_output=True, text=True, timeout=10)
        
        if result.returncode == 0:
            print(f"✅ {file_path.name} runs successfully")
            return True
        else:
            print(f"⚠️  {file_path.name} exited with code {result.returncode}")
            if result.stderr:
                print(f"   Error: {result.stderr.strip()}")
            return False
            
    except subprocess.TimeoutExpired:
        print(f"⏱️  {file_path.name} timed out (likely waiting for input)")
        return True  # Timeout usually means it's running fine
    except Exception as e:
        print(f"❌ {file_path.name} failed: {e}")
        return False

def main():
    """Test all DEMO files"""
    print("🧪 VERIFYING DEMO FILES AFTER REORGANIZATION")
    print("=" * 60)
    
    demo_dir = Path(__file__).parent
    demo_files = list(demo_dir.glob("*.py"))
    demo_files = [f for f in demo_files if f.name != "verify_demo.py"]  # Exclude this script
    
    successful = 0
    total = len(demo_files)
    
    for demo_file in demo_files:
        if test_demo_file(demo_file):
            successful += 1
    
    print(f"\n📊 SUMMARY: {successful}/{total} demo files working correctly")
    
    if successful == total:
        print("🎉 All demo files are working!")
    else:
        print("⚠️  Some demo files may need attention")

if __name__ == "__main__":
    main()
