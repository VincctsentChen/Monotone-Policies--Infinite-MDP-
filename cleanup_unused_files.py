#!/usr/bin/env python3
"""
Script to safely delete unused files while preserving the working test_infinite_horizon_with_data.py
"""

import os
import shutil

def cleanup_unused_files():
    """Delete unused files while preserving the working test script"""
    
    # Files to delete (unused by test_infinite_horizon_with_data.py)
    files_to_delete = [
        # Finite horizon files
        "hypertension_treatment_monotone_mdp.py",
        "hypertension_treatment_infinite_mdp.py", 
        "patient_simulation.py",
        "optimal_monotone_mdp.py",
        "optimal_monotone_mdp_infinite.py",
        "policy_evaluation.py",
        "transition_probabilities.py",
        "aha_2017_guideline.py",
        "risk_based_policy.py",
        
        # Other test files
        "test_imports.py",
        "test_infinite_horizon_simple.py",
        "test_infinite_horizon.py",
        "test_minimal_infinite.py",
        
        # Analysis/plotting files
        "case_study_plots.py",
        "case_study_results.py",
        "example_plot.py",
        "weighted_quantiles.py",
        
        # Documentation
        "INFINITE_HORIZON_CONVERSION_GUIDE.md",
        "README.md",
        
        # This cleanup script itself
        "cleanup_unused_files.py"
    ]
    
    # Directories to delete
    dirs_to_delete = [
        "__pycache__",
        "Figures",
        "Results"
    ]
    
    print("🧹 Cleaning up unused files...")
    print("=" * 50)
    
    deleted_files = []
    deleted_dirs = []
    errors = []
    
    # Delete files
    for file in files_to_delete:
        if os.path.exists(file):
            try:
                os.remove(file)
                deleted_files.append(file)
                print(f"✅ Deleted file: {file}")
            except Exception as e:
                errors.append(f"❌ Error deleting {file}: {e}")
        else:
            print(f"⚠️  File not found: {file}")
    
    # Delete directories
    for dir_name in dirs_to_delete:
        if os.path.exists(dir_name):
            try:
                shutil.rmtree(dir_name)
                deleted_dirs.append(dir_name)
                print(f"✅ Deleted directory: {dir_name}")
            except Exception as e:
                errors.append(f"❌ Error deleting {dir_name}: {e}")
        else:
            print(f"⚠️  Directory not found: {dir_name}")
    
    print("\n" + "=" * 50)
    print("📊 Cleanup Summary:")
    print(f"   - Files deleted: {len(deleted_files)}")
    print(f"   - Directories deleted: {len(deleted_dirs)}")
    print(f"   - Errors: {len(errors)}")
    
    if errors:
        print("\n❌ Errors encountered:")
        for error in errors:
            print(f"   {error}")
    
    print("\n✅ Remaining files (required for test_infinite_horizon_with_data.py):")
    remaining_files = [
        "test_infinite_horizon_with_data.py",
        "patient_simulation_infinite_no_gurobi.py",
        "ascvd_risk.py",
        "transition_probabilities_infinite.py",
        "policy_evaluation_infinite.py",
        "aha_2017_guideline_infinite.py",
        "risk_based_policy_infinite.py",
        "post_treatment_risk.py",
        "sbp_reductions_drugtype.py",
        "dbp_reductions_drugtype.py",
        "Data/ (and all CSV files inside)"
    ]
    
    for file in remaining_files:
        print(f"   📁 {file}")
    
    print(f"\n🎯 Total files remaining: {len(remaining_files)}")
    print("🚀 Your directory is now clean and optimized!")

if __name__ == "__main__":
    cleanup_unused_files()
