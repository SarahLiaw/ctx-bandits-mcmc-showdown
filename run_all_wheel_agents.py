#!/usr/bin/env python3
"""
Script to run all MCMC bandit agents on the wheel dataset.

This script will run each agent with the specified number of trials and save the results.
"""
import os
import subprocess
import argparse
import json
import time
from pathlib import Path
from tqdm import tqdm

def run_agent(agent_name, num_trials=1, gpu_id=None, trial_offset=0):
    """Run a single agent with the specified number of trials.
    
    Args:
        agent_name: Name of the agent to run
        num_trials: Number of trials to run
        gpu_id: GPU ID to use (if any)
        trial_offset: Offset to ensure unique seeds across different agents
    """
    import time
    from pathlib import Path
    
    # Set up environment variables
    env = os.environ.copy()
    if gpu_id is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    
    # Get absolute path to config
    config_path = Path(f"config/wheel/{agent_name.lower()}.json").absolute()
    
    print(f"Running {agent_name} with {num_trials} trials")
    
    # Run each trial with a different seed
    for trial in range(num_trials):
        # Generate a unique seed for this agent and trial
        # Using a combination of agent name, trial number, and current time
        seed = hash(agent_name) % (2**32 - 1) + (trial + 1) * 1000 + int(time.time()) % 1000
        
        # Build the command - run.py only accepts --config_path
        cmd = [
            "python3", "run.py",
            f"--config_path={config_path}"
        ]
        
        # Set the seed via environment variable
        env["PYTHONHASHSEED"] = str(seed)
        
        print(f"\nTrial {trial+1}/{num_trials}")
        print(f"Agent: {agent_name}")
        print(f"Seed: {seed}")
        print(f"Command: {' '.join(cmd)}")
        
        try:
            # Run with a timeout to prevent hanging
            start_time = time.time()
            result = subprocess.run(
                cmd,
                env=env,
                check=True,
                timeout=600,  # 10 minute timeout
                capture_output=True,
                text=True
            )
            
            # Print output
            if result.stdout:
                print("\n".join(result.stdout.split('\n')[-10:]))  # Show last 10 lines of output
            
            if result.stderr:
                print("Errors:", result.stderr)
                
            print(f"Completed in {time.time() - start_time:.2f} seconds")
            
        except subprocess.TimeoutExpired:
            print(f"\nTrial {trial+1} timed out after 10 minutes. Skipping...")
            continue
            
        except subprocess.CalledProcessError as e:
            print(f"\nError running {agent_name} trial {trial+1}:")
            if hasattr(e, 'stdout') and e.stdout:
                stdout = e.stdout.decode() if hasattr(e.stdout, 'decode') else e.stdout
                print("\n".join(stdout.split('\n')[-20:]))
            if hasattr(e, 'stderr') and e.stderr:
                stderr = e.stderr.decode() if hasattr(e.stderr, 'decode') else e.stderr
                print("Errors:", stderr)
            continue
            
        except KeyboardInterrupt:
            print("\nReceived keyboard interrupt. Cleaning up...")
            # Perform any necessary cleanup here
            raise  # Re-raise to allow proper termination
            
        except Exception as e:
            print(f"\nUnexpected error in trial {trial+1}: {str(e)}")
            import traceback
            traceback.print_exc()
            continue

def main():
    parser = argparse.ArgumentParser(description='Run all MCMC bandit agents on the wheel dataset')
    parser.add_argument('--num_trials', type=int, default=1, 
                       help='Number of trials to run for each agent')
    parser.add_argument('--gpu_id', type=int, default=None, 
                       help='GPU ID to use (if any). Example: 0,1,2')
    parser.add_argument('--agents', nargs='+', default=None,
                       help='List of agents to run (default: run all)')
    parser.add_argument('--output_dir', type=str, default='results',
                       help='Directory to save results')
    args = parser.parse_args()
    
    # List of all available agents
    all_agents = [
        'lmcts', 'fglmc', 'malats', 'fgmalats', 'sfglmc', 'sfgmala',
        'plmc', 'pfglmc', 'psfglmc', 'svrglmc', 'ulmc', 'ulmcfgts', 'ulmcsfgts'
    ]
    
    # Determine which agents to run
    agents_to_run = args.agents if args.agents is not None else all_agents
    agents_to_run = [a.lower() for a in agents_to_run]  # Normalize case
    
    # Validate agents
    invalid_agents = set(agents_to_run) - set(all_agents)
    if invalid_agents:
        print(f"Warning: Unknown agents: {', '.join(invalid_agents)}. They will be skipped.")
        agents_to_run = [a for a in agents_to_run if a in all_agents]
    
    if not agents_to_run:
        print("No valid agents to run. Exiting.")
        return
    
    # Create output directory if it doesn't exist
    out_dir = Path(args.output_dir).absolute()
    out_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*80}")
    print(f"Starting experiment with {len(agents_to_run)} agents and {args.num_trials} trials each")
    print(f"Agents: {', '.join(agents_to_run)}")
    print(f"Output directory: {out_dir}")
    
    # Track successful and failed runs
    results = {
        'successful': {},
        'failed': {}
    }
    
    start_time = time.time()
    
    try:
        # Run each agent with a different trial_offset to ensure unique seeds
        for i, agent in enumerate(agents_to_run):
            agent_start = time.time()
            print(f"\n{'*'*40}")
            print(f"Starting agent: {agent} ({i+1}/{len(agents_to_run)})")
            
            try:
                run_agent(agent, args.num_trials, args.gpu_id, trial_offset=i)
                results['successful'][agent] = {
                    'trials': args.num_trials,
                    'time_seconds': time.time() - agent_start
                }
                print(f"\n✅ Completed {agent} in {time.time() - agent_start:.2f} seconds")
            except Exception as e:
                results['failed'][agent] = {
                    'error': str(e),
                    'time_seconds': time.time() - agent_start
                }
                print(f"\n❌ Error in {agent}: {str(e)}")
                import traceback
                traceback.print_exc()
    
    except KeyboardInterrupt:
        print("\n" + "="*60)
        print("Experiment interrupted by user. Saving current results...")
    
    # Print summary
    total_time = time.time() - start_time
    print("\n" + "="*60)
    print("EXPERIMENT SUMMARY")
    print("="*60)
    print(f"Total time: {total_time/60:.1f} minutes")
    print(f"Successful agents ({len(results['successful'])}): {', '.join(results['successful'].keys()) or 'None'}")
    
    if results['failed']:
        print(f"\nFailed agents ({len(results['failed'])}):")
        for agent, info in results['failed'].items():
            print(f"  - {agent}: {info['error']}")
    
    # Save results summary
    summary = {
        'start_time': time.strftime('%Y-%m-%d %H:%M:%S'),
        'total_duration_seconds': total_time,
        'successful_agents': len(results['successful']),
        'failed_agents': len(results['failed']),
        'agents': {
            'successful': results['successful'],
            'failed': results['failed']
        }
    }
    
    summary_file = out_dir / 'experiment_summary.json'
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\nSummary saved to: {summary_file}")
    print("Experiment completed!")

if __name__ == '__main__':
    main()
