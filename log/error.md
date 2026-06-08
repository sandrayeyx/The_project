(.venv) PS E:\project\code\algo-qi> .\run_pipeline.ps1
[GPU] full project pipeline: torch=2.11.0+cu130, device=cuda:0 (NVIDIA GeForce RTX 4070)

[SIMULATION] E:\project\code\algo-qi\.venv\Scripts\python.exe E:\project\code\algo-qi\src\iterative_testing\iterative_failure_simulation.py --config E:\project\code\algo-qi\data\tmp_data\full_project_runs\20260604_205702\runtime_config.yaml --env-md E:\project\code\algo-qi\config\environment\env_config.md --exploration-config E:\project\code\algo-qi\src\failure_and_attribution_analysis\config\scenario_exploration.yaml --output-root E:\project\code\algo-qi\data\tmp_data\full_project_runs\20260604_205702\simulation_data\2\20260604_205702 --raw-log-root E:\project\code\algo-qi\data\tmp_data\full_project_runs\20260604_205702\simulation_data\2\20260604_205702 --generated-limit 500 --scenarios-per-round 16 --seed-per-region 48 --coverage-target 0.9 --min-samples-for-coverage-stop 100 --stop-on-coverage-target true --true-failure-policy strict --failure-decision-mode single_fused_score --fused-model-type mlp_small --fit-decision-model-offline true --threshold-calibration-scope terminal_only --threshold-calibration-mode two_stage_stable --allow-multi-attacks-per-scenario false --single-attack-types StateObservationAttack,ModelTampAttack --online-backfill-after-each-round true --post-run-offline-recompute true --enable-accuracy-guard true --min-failure-detection-accuracy 0.9 --reset-state
[GPU] closed-loop failure simulation: torch=2.11.0+cu130, device=cuda:0 (NVIDIA GeForce RTX 4070)
C:\Users\a\AppData\Local\Programs\Python\Python311\python.exe: can't open file 'E:\\project\\code\\algo-qi\\iterative_testing\\PRC.py': [Errno 2] No such file or directory
Traceback (most recent call last):
  File "E:\project\code\algo-qi\src\iterative_testing\iterative_failure_simulation.py", line 5234, in <module>
    raise SystemExit(main())
                     ^^^^^^
  File "E:\project\code\algo-qi\src\iterative_testing\iterative_failure_simulation.py", line 5226, in main
    workflow.run()
  File "E:\project\code\algo-qi\src\iterative_testing\iterative_failure_simulation.py", line 5158, in run
    round_summary_records, round_step_records = self._collect_round_results(self.round_index, current_scenarios)
                                                ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "E:\project\code\algo-qi\src\iterative_testing\iterative_failure_simulation.py", line 3026, in _collect_round_results
    self._run_single_simulation(temp_config_path)
  File "E:\project\code\algo-qi\src\iterative_testing\iterative_failure_simulation.py", line 1378, in _run_single_simulation
    raise RuntimeError(f"PRC.py failed with exit code {completed.returncode} for {temp_config_path.name}")
RuntimeError: PRC.py failed with exit code 2 for config.yaml
Traceback (most recent call last):
  File "E:\project\code\algo-qi\run_full_project_pipeline.py", line 1875, in <module>
    raise SystemExit(main())
                     ^^^^^^
  File "E:\project\code\algo-qi\run_full_project_pipeline.py", line 1711, in main
    run_command(
  File "E:\project\code\algo-qi\run_full_project_pipeline.py", line 1005, in run_command
    raise subprocess.CalledProcessError(return_code, list(cmd))
subprocess.CalledProcessError: Command '['E:\\project\\code\\algo-qi\\.venv\\Scripts\\python.exe', 'E:\\project\\code\\algo-qi\\src\\iterative_testing\\iterative_failure_simulation.py', '--config', 'E:\\project\\code\\algo-qi\\data\\tmp_data\\full_project_runs\\20260604_205702\\runtime_config.yaml', '--env-md', 'E:\\project\\code\\algo-qi\\config\\environment\\env_config.md', '--exploration-config', 'E:\\project\\code\\algo-qi\\src\\failure_and_attribution_analysis\\config\\scenario_exploration.yaml', '--output-root', 'E:\\project\\code\\algo-qi\\data\\tmp_data\\full_project_runs\\20260604_205702\\simulation_data\\2\\20260604_205702', '--raw-log-root', 'E:\\project\\code\\algo-qi\\data\\tmp_data\\full_project_runs\\20260604_205702\\simulation_data\\2\\20260604_205702', '--generated-limit', '500', '--scenarios-per-round', '16', '--seed-per-region', '48', '--coverage-target', '0.9', '--min-samples-for-coverage-stop', '100', '--stop-on-coverage-target', 'true', '--true-failure-policy', 'strict', '--failure-decision-mode', 'single_fused_score', '--fused-model-type', 'mlp_small', '--fit-decision-model-offline', 'true', '--threshold-calibration-scope', 'terminal_only', '--threshold-calibration-mode', 'two_stage_stable', '--allow-multi-attacks-per-scenario', 'false', '--single-attack-types', 'StateObservationAttack,ModelTampAttac> (Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned) ; (& e:\project\code\algo-qi\.venv\Scripts\Activate.ps1)'--min-failure-detection-accuracy', '0.9', '--reset-state']' returned non-zero exit status 1.