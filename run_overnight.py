import subprocess, os

videos = ['18Promont', '8KTMtoLFPS', '30MysRoad', '7TajaBakery', '13UnionBankHsk']

for v in videos:
    env = {**os.environ,
           "URSA_VIDEO": v,
           "URSA_OUTPUT_DIR": f"outputs/{v}"}
    
    print(f"\n>>> Extracting frames: {v}")
    subprocess.run(["python", "extract_frames.py"], env=env)
    subprocess.run(["python", "frame_reliability.py"], env=env)
    subprocess.run(["python", "reliability_filter.py"], env=env)
    
    print(f"\n>>> TTA: {v}")
    subprocess.run(["python", "uncertainty_estimation_tta.py"], env=env)
    
    print(f"\n>>> MCD: {v}")
    subprocess.run(["python", "uncertainty_estimation_mcd.py"], env=env)
    
    # cleanup to save disk
    import shutil
    shutil.rmtree(f"outputs/{v}/frames", ignore_errors=True)
    shutil.rmtree(f"outputs/{v}/filtered_frames", ignore_errors=True)
    print(f">>> Cleaned up frames for {v}")

print("\nAll done.")