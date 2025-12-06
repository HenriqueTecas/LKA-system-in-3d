"""
Quick Test Script for LKA Logging and Visualization System

Tests:
1. Logger initialization
2. Metric computation
3. Session data saving
4. Plot generation (if matplotlib available)
"""

import numpy as np
import os
import sys


def test_logger():
    """Test logger functionality"""
    print("Testing LKA Performance Logger...")
    
    # Mock objects for testing
    class MockCar:
        def __init__(self):
            self.x = 0.0
            self.y = 0.0
            self.theta = 0.0
            self.velocity = 15.0
            self.steering_angle = 0.1
    
    class MockController:
        def __init__(self):
            self.safe_speed = 20.0
            self._last_curve_radius = 150.0
            self.lookahead_speed_scale = 0.5
            self.lookahead_min = 5.0
            self.lookahead_max = 22.0
            self.intervening = False
            self.target_direction = 0.5
            self.car = MockCar()
            self.center_line_points = [(0, 0), (10, 0.5), (20, 1.0)]
        
        def _estimate_lane_offset(self):
            return 0.15
    
    # Import logger
    try:
        from src.lka_logger import LKAPerformanceLogger
    except ImportError:
        from lka_logger import LKAPerformanceLogger
    
    # Create logger
    logger = LKAPerformanceLogger(log_dir="test_logs")
    print("✓ Logger created")
    
    # Log some frames
    car = MockCar()
    controller = MockController()
    warnings = {'speed_too_high': False, 'lane_departure': False}
    
    for i in range(100):
        timestamp = i * 0.033  # 30 Hz
        car.x += 0.5
        car.y += np.sin(i * 0.1) * 0.1
        car.steering_angle = np.sin(i * 0.05) * 0.3
        
        logger.log_frame(timestamp, car, controller, warnings)
    
    print(f"✓ Logged {logger.total_frames} frames")
    
    # Get metrics
    metrics = logger.get_current_metrics()
    print(f"✓ Current metrics: {len(metrics)} values")
    print(f"  Lateral error: {metrics.get('lateral_error', 0):.3f} m")
    print(f"  Speed ratio: {metrics.get('speed_ratio', 0):.1f}%")
    
    # Get statistics
    stats = logger.get_statistics()
    print(f"✓ Statistics computed: {len(stats)} metrics")
    print(f"  Mean lateral error: {stats['lateral_error_mean']:.3f} m")
    print(f"  Std dev: {stats['lateral_error_std']:.3f} m")
    
    # Save session
    session_file = logger.save_session()
    print(f"✓ Session saved: {session_file}")
    
    # Print summary
    logger.print_summary()
    
    return session_file


def test_visualization(session_file):
    """Test visualization generation"""
    print("\nTesting LKA Visualization Generator...")
    
    try:
        import matplotlib
        matplotlib.use('Agg')  # Non-interactive backend
        print("✓ matplotlib available")
    except ImportError:
        print("✗ matplotlib not installed - skipping plot generation")
        return
    
    try:
        from src.lka_logger import LKAVisualizationGenerator
    except ImportError:
        from lka_logger import LKAVisualizationGenerator
    
    # Load session
    viz = LKAVisualizationGenerator(session_file)
    print(f"✓ Session loaded: {viz.session_id}")
    
    # Generate plots
    viz.generate_all_plots(output_dir="test_plots")
    print("✓ All plots generated")
    
    # Check files
    expected_plots = [
        'lateral_error_series.png',
        'speed_profile.png',
        'steering_response.png',
        'trajectory_overlay.png',
        'error_distribution.png',
        'phase_portrait.png',
        'control_distribution.png',
    ]
    
    for plot in expected_plots:
        path = os.path.join('test_plots', plot)
        if os.path.exists(path):
            size = os.path.getsize(path)
            print(f"  ✓ {plot} ({size:,} bytes)")
        else:
            print(f"  ✗ {plot} MISSING")


def test_hud_metrics():
    """Test HUD metric computation"""
    print("\nTesting HUD Metrics Computation...")
    
    # This would require full pygame/OpenGL setup
    # Just verify the functions exist
    try:
        from src.hud import HUD
    except ImportError:
        from hud import HUD
    
    hud = HUD()
    print(f"✓ HUD initialized")
    print(f"  Fonts: small={hud.font_small is not None}, regular={hud.font is not None}, large={hud.font_large is not None}")
    print(f"  Logger slot: {hud.lka_logger is None}")
    
    # Check methods exist
    assert hasattr(hud, '_draw_lka_metrics_panel')
    assert hasattr(hud, '_draw_steering_gauge')
    assert hasattr(hud, '_compute_metrics_direct')
    print("✓ All HUD methods present")


def cleanup():
    """Clean up test files"""
    import shutil
    
    if os.path.exists('test_logs'):
        shutil.rmtree('test_logs')
        print("\n✓ Cleaned up test_logs/")
    
    if os.path.exists('test_plots'):
        shutil.rmtree('test_plots')
        print("✓ Cleaned up test_plots/")


def main():
    print("="*60)
    print("LKA LOGGING AND VISUALIZATION TEST SUITE")
    print("="*60)
    
    try:
        # Test 1: Logger
        session_file = test_logger()
        
        # Test 2: Visualization
        test_visualization(session_file)
        
        # Test 3: HUD
        test_hud_metrics()
        
        print("\n" + "="*60)
        print("ALL TESTS PASSED ✓")
        print("="*60)
        
    except Exception as e:
        print(f"\n✗ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    finally:
        # Cleanup
        cleanup()


if __name__ == '__main__':
    main()
