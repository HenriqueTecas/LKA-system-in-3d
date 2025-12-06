# LKA HUD & Logging Quick Reference

## HUD Metrics Panel (Right Side)

```
┌─────────────────────────────────┐
│      LKA METRICS               │
├─────────────────────────────────┤
│ Lateral Error:    +0.123 m  🟢 │  Color: 🟢<0.5m 🟡0.5-1.0m 🔴>1.0m
│ Heading Error:    -2.3°        │  Angular deviation from path
│ Speed Ratio:      85%       🟡 │  v/v_safe percentage
│ [████████░░░░░░░░░░░░░░░░░]    │  Visual speed bar
│ Curve Radius:     STRAIGHT  🟢 │  "STRAIGHT" if R>500m
│ Lookahead:        12.5 m       │  Adaptive 5-22m range
│                                 │
│      Steering Gauge             │
│          -35°   0°   +35°       │
│            ╲    │    ╱          │
│             ╲   │   ╱           │
│              ╲  │  ╱            │
│               ╲ │ ╱             │
│                \│/              │
│                 ●───→           │  Needle shows current angle
│              +12.5°          🟢 │  Color: 🟢<25° 🟡25-34° 🔴≥34°
│                                 │
│         [ CENTERED ]         🟢 │  State: CENTERED/ASSISTING
└─────────────────────────────────┘
```

## Key Controls

| Key | Action |
|-----|--------|
| `1` | Manual mode (no assist) |
| `2` | Warning mode (alerts only) |
| `3` | Assist mode (active control) |
| `L` | **Save session & generate plots** |
| `C` | Toggle camera view |
| `ESC` | Exit (auto-saves) |

## Quick Actions

### View Real-Time Metrics
1. Press `2` or `3` to activate LKA
2. HUD panel appears on right side
3. Watch color-coded metrics

### Save Performance Data
1. While running, press `L`
2. Waits 2-5 seconds for plot generation
3. Check `logs/` and `plots/` folders
4. Console shows summary statistics

### Generate Plots Later
```bash
python -m src.generate_plots --latest
```

## Color Codes

### Lateral Error
- 🟢 **Green** (< 0.5m): Perfect tracking
- 🟡 **Yellow** (0.5-1.0m): Minor deviation
- 🔴 **Red** (> 1.0m): Needs correction

### Speed Ratio
- 🟢 **Green** (< 80%): Safe margin
- 🟡 **Yellow** (80-100%): Near limit
- 🔴 **Red** (> 100%): Overspeed!

### Steering Angle
- 🟢 **Green** (< 25°): Normal
- 🟡 **Yellow** (25-34°): High
- 🔴 **Red** (≥ 34°): Saturated

## Generated Plots

When you press `L` or exit:

1. `lateral_error_series.png` - Error over time
2. `speed_profile.png` - Speed vs safe speed
3. `steering_response.png` - Control response
4. `trajectory_overlay.png` - 2D path view
5. `error_distribution.png` - Statistical analysis
6. `phase_portrait.png` - Stability visualization
7. `control_distribution.png` - Steering histogram

## Console Output Example

```
[LKA Logger] Saving session data...
============================================================
LKA PERFORMANCE SUMMARY
============================================================
Session Duration: 147.3 s
Total Frames: 4419

LATERAL TRACKING:
  Mean Error: +0.087 m
  Std Dev: 0.124 m
  Max Error: 0.456 m

SPEED CONTROL:
  Compliance Rate: 97.3%
  Speed Violations: 23

INTERVENTIONS:
  Total Count: 12
  Frequency: 4.88 /min
  Mean Duration: 0.234 s

STEERING:
  Mean Angle: -1.2°
  Std Dev: 8.7°
============================================================

Session data saved to: logs/lka_session_20231206_143022.json
[LKA Logger] ✓ Plots saved to plots/ directory
```

## What Each Metric Means

**Lateral Error**
- Distance from lane center (perpendicular)
- Positive = right of center
- Negative = left of center

**Heading Error**
- Angle difference from desired direction
- Positive = pointing right
- Negative = pointing left

**Speed Ratio**
- Current speed ÷ safe speed
- 100% = at safe speed limit
- > 100% = exceeding safe speed

**Curve Radius**
- Tightness of current curve
- STRAIGHT = R > 500m (gentle/straight)
- < 100m = tight curve
- < 50m = very sharp

**Lookahead**
- How far ahead system is planning
- Increases with speed (0.5s preview)
- Decreases in curves
- Range: 5-22m

**Steering Angle**
- Current wheel angle
- Left = negative
- Right = positive
- Limit = ±35°

**Intervention State**
- CENTERED = within tolerances (green zone)
- ASSISTING = active correction (outside green zone)

## File Locations

```
logs/
  lka_session_20231206_143022.json  ← Session data

plots/
  lateral_error_series.png          ← 7 plots here
  speed_profile.png
  steering_response.png
  trajectory_overlay.png
  error_distribution.png
  phase_portrait.png
  control_distribution.png
```

## Tips

**Best Practices:**
- Activate mode `3` (ASSIST) for full logging
- Drive for 2-3 minutes minimum
- Include various scenarios (straight, curves)
- Press `L` before exiting for best plots

**Performance:**
- Logging overhead: < 1% CPU
- HUD panel: < 2 FPS impact
- Plot generation: 2-5 seconds

**Troubleshooting:**
- Metrics not showing? Check mode is `2` or `3` (not `1`)
- Plots not generated? Install: `pip install matplotlib scipy`
- Empty logs? Make sure LKA mode is active

## Reading the Statistics

**Good Performance:**
- Lateral error std < 0.15m
- Speed compliance > 95%
- Max error < 1.0m
- Intervention duration < 1.0s

**Excellent Performance:**
- Lateral error std < 0.10m
- Speed compliance > 99%
- Max error < 0.6m
- Intervention duration < 0.3s

## See Full Documentation

- `LKA_LOGGING_README.md` - Complete guide
- `LKA_ENHANCEMENT_SUMMARY.md` - Technical details
- `LKA_DOCUMENTATION.tex` - Mathematical foundation
