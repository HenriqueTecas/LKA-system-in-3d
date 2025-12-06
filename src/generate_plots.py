"""
Generate LKA Performance Plots from Logged Session

Usage:
    python -m src.generate_plots <session_file.json>
    
Or to process the latest session:
    python -m src.generate_plots --latest
"""

import sys
import os
import glob
import argparse


def main():
    parser = argparse.ArgumentParser(description='Generate LKA performance visualization plots')
    parser.add_argument('session_file', nargs='?', help='Path to session JSON file')
    parser.add_argument('--latest', action='store_true', help='Use the latest session file')
    parser.add_argument('--output', '-o', default='plots', help='Output directory for plots')
    
    args = parser.parse_args()
    
    # Determine session file
    if args.latest:
        # Find latest session file
        session_files = glob.glob('logs/lka_session_*.json')
        if not session_files:
            print("ERROR: No session files found in logs/ directory")
            sys.exit(1)
        session_file = max(session_files, key=os.path.getctime)
        print(f"Using latest session: {session_file}")
    elif args.session_file:
        session_file = args.session_file
        if not os.path.exists(session_file):
            print(f"ERROR: Session file not found: {session_file}")
            sys.exit(1)
    else:
        print("ERROR: Please provide a session file or use --latest")
        parser.print_help()
        sys.exit(1)
    
    # Import and generate plots
    try:
        from .lka_logger import LKAVisualizationGenerator
    except ImportError:
        # Try direct import if running as script
        from lka_logger import LKAVisualizationGenerator
    
    print(f"Loading session data from: {session_file}")
    viz = LKAVisualizationGenerator(session_file)
    
    print(f"Generating plots to: {args.output}/")
    viz.generate_all_plots(output_dir=args.output)
    
    print("\n✓ All plots generated successfully!")
    print(f"  Check: {args.output}/")


if __name__ == '__main__':
    main()
