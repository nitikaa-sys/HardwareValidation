#!/usr/bin/env python3
"""
ADS1299 Experiment Runner - CLI Entry Point

Thin wrapper that orchestrates experiment execution using the acquisition modules.

Usage:
    python run_experiment.py                              # Default: loads config.json
    python run_experiment.py --config my_experiment.json  # Custom config
    python run_experiment.py --events 10                  # Override num_events
    python run_experiment.py --events 50 --ip 192.168.1.100 --profile profiles/all_shorted.json
"""

import asyncio
import argparse
import sys
from pathlib import Path

# Add parent directory to path for module imports when run directly
SCRIPT_DIR = Path(__file__).parent.resolve()
sys.path.insert(0, str(SCRIPT_DIR.parent))

from acquisition import (
    HardwareClient,
    load_config,
    validate_experiment_config,
    merge_config_with_args,
    get_default_config_path,
    setup_random_seed,
    generate_event_schedule,
    display_event_schedule,
    run_config_experiment,
    run_experiment
)
from acquisition.constants import DEFAULT_ESP32_IP


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description='ADS1299 Experiment Runner',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  # Default: loads config.json automatically
  %(prog)s
  
  # Use custom config file
  %(prog)s --config my_experiment.json
  
  # Override specific config values
  %(prog)s --events 20                     # Override num_events
  %(prog)s --events 50 --ip 192.168.1.100  # Override events + ESP32 IP
  %(prog)s --profile profiles/all_shorted.json   # Override ADS1299 profile

Configuration:
  - By default, loads ./config.json
  - CLI flags (--events, --ip, --profile) override config values
  - Supports balanced experimental design with reproducible random seeds
  - Falls back to simple CLI mode if config.json missing and --events provided
        '''
    )
    
    parser.add_argument('-c', '--config', type=str, default=None,
                       help='Path to config file (default: config.json)')
    parser.add_argument('-n', '--events', type=int, default=None,
                       help='Number of events (overrides config value)')
    parser.add_argument('-i', '--ip', '--host', dest='ip', type=str, default=DEFAULT_ESP32_IP,
                       help=f'ESP32 IP or hostname (overrides config, default: {DEFAULT_ESP32_IP})')
    parser.add_argument('-p', '--profile', type=str, default=None,
                       help='ADS1299 profile JSON file (overrides config value)')
    
    return parser.parse_args()


def main() -> int:
    """Main entry point."""
    args = parse_arguments()
    
    # Determine config path
    if args.config:
        config_path = Path(args.config)
    else:
        config_path = get_default_config_path()
    
    try:
        # ============ STEP 1: Load and validate configuration ============
        print(f"\n🔧 ADS1299 Experiment Runner")
        print(f"=" * 70)
        print(f"\n[STEP 1/5] Loading configuration...")
        
        config = load_config(str(config_path))
        config = merge_config_with_args(config, args)
        validate_experiment_config(config)
        
        # Create hardware client
        esp_ip = config['esp32']['ip']
        client = HardwareClient(esp_ip)
        
        num_events = config['experiment']['num_events']
        profile_path = config['experiment'].get('profile_path', None)
        
        # ============ STEP 2: Setup reproducibility ============
        print(f"\n[STEP 2/5] Setting up reproducibility...")
        setup_random_seed(config)
        
        # ============ STEP 3: Generate event schedule ============
        print(f"\n[STEP 3/5] Generating event schedule...")
        schedule = generate_event_schedule(
            num_events,
            config['conditions']['mapping'],
            config['conditions']['enforce_equal_condition_count']
        )
        display_event_schedule(
            schedule,
            config['conditions']['enforce_equal_condition_count'],
            config['experiment'].get('random_seed')
        )
        
        # ============ STEP 4: Load profile if specified ============
        if profile_path:
            print(f"\n[STEP 4/5] Loading ADS1299 profile...")
            if not client.load_and_apply_profile(profile_path):
                print("\n✗ EXPERIMENT ABORTED: Profile loading failed")
                return 1
            asyncio.run(asyncio.sleep(0.5))
        else:
            print(f"\n[STEP 4/5] Skipping profile load (none specified)")
        
        # ============ STEP 5: Verify registers and get user confirmation ============
        print(f"\n[STEP 5/5] Verifying ADS1299 configuration...")
        registers = client.dump_and_verify_registers()
        if not registers:
            print("\n✗ EXPERIMENT ABORTED: Could not read registers")
            return 1
        
        print("\n" + "="*70)
        print("Please review the configuration, event schedule, and register dump above.")
        user_input = input("Continue with experiment? [y/N]: ").strip().lower()
        if user_input != 'y':
            print("\n✗ EXPERIMENT ABORTED: User cancelled")
            return 0
        
        # ============ Run experiment ============
        print(f"\n" + "="*70)
        print(f"STARTING CONFIG-DRIVEN EXPERIMENT")
        print(f"=" * 70)
        print(f"📋 Configuration: {config_path}")
        print(f"🎲 Random seed: {config['experiment'].get('random_seed', 'none')}")
        print(f"⚖️  Equal counts: {config['conditions']['enforce_equal_condition_count']}")
        print(f"🎯 Events: {num_events}")
        print(f"=" * 70)
        
        # Run experiment with hardware setup already done
        success = asyncio.run(run_config_experiment(
            config=config,
            schedule=schedule,
            client=client,
            skip_prompts=False,
            status_cb=None,
            stop_flag=None,
            skip_hardware_setup=True  # Already done in Steps 4/5
        ))
        
        return 0 if success else 1
        
    except ValueError as e:
        # Config file issue - try legacy fallback
        if not args.config and args.events is not None:
            # Fall back to legacy mode
            print(f"\n🔧 ADS1299 Experiment Runner (Legacy Mode)")
            print(f"  --events: {args.events}")
            print(f"  --ip: {args.ip}")
            
            client = HardwareClient(args.ip)
            
            try:
                success = asyncio.run(run_experiment(
                    args.events,
                    client,
                    args.profile
                ))
                return 0 if success else 1
            except KeyboardInterrupt:
                print("\n\n✗ Interrupted")
                return 1
            except Exception as e:
                print(f"\n✗ Error: {e}")
                return 1
        else:
            print(f"\n✗ CONFIG ERROR: {e}")
            if not args.config:
                print(f"   Create config.json or use: python run_experiment.py --events NUM")
            return 1
    
    except KeyboardInterrupt:
        print("\n\n✗ Experiment interrupted by user")
        return 1
    except Exception as e:
        print(f"\n✗ Fatal error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
