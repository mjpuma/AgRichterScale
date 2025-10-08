# H-P Envelope Axis Verification

## Question
Is there confusion about plotting magnitude (1-7) vs 10^1 to 10^7 in the H-P envelope?

## Answer: The Current Implementation is Correct ✅

### X-Axis: Magnitude (M_D)

**What is plotted:** Magnitude values (2, 3, 4, 5, 6, 7)

**How it's calculated:**
```python
# From hp_envelope.py line 60
magnitude = np.log10(disrupted_areas)  # disrupted_areas is in km²
```

**What it represents:**
- Magnitude 2 = 10² km² = 100 km²
- Magnitude 3 = 10³ km² = 1,000 km²
- Magnitude 4 = 10⁴ km² = 10,000 km²
- Magnitude 5 = 10⁵ km² = 100,000 km²
- Magnitude 6 = 10⁶ km² = 1,000,000 km²
- Magnitude 7 = 10⁷ km² = 10,000,000 km²

**Axis configuration:**
```python
# From hp_envelope.py line 63
ax.set_xlim([2, 7])  # Magnitude range

# From hp_envelope.py line 85
ax.set_xlabel('Magnitude M_D = log₁₀(A_H) [km²]', fontsize=12)
```

**Scale:** Linear scale (showing magnitude values 2, 3, 4, 5, 6, 7)

---

### Y-Axis: Production Loss (kcal)

**What is plotted:** Actual production loss values in kcal

**Values:**
```python
# From hp_envelope.py lines 61-62
upper_bound = np.array(envelope_data['upper_bound'])  # In kcal
lower_bound = np.array(envelope_data['lower_bound'])  # In kcal
```

**Axis configuration:**
```python
# From hp_envelope.py line 64
ax.set_ylim([1e10, 1.62e16])  # Actual kcal values

# From hp_envelope.py line 82
ax.set_yscale('log')  # Logarithmic scale

# From hp_envelope.py line 86
ax.set_ylabel('Production Loss [kcal]', fontsize=12)
```

**Scale:** Logarithmic scale (showing 10^10, 10^11, 10^12, ..., 10^16)

---

## Why This is Correct

### The H-P Envelope Plot Shows:

1. **X-axis (Magnitude):** 
   - Already log-transformed values (2-7)
   - Represents log₁₀(harvest area in km²)
   - Linear scale showing magnitude values

2. **Y-axis (Production Loss):**
   - Actual kcal values (not log-transformed)
   - Range: 10^10 to 10^16 kcal
   - Logarithmic scale for display (so we see 10^10, 10^11, etc.)

### This Matches the AgriRichter Specification

From the original MATLAB code and specification:
- Magnitude is defined as: **M_D = log₁₀(A_H)** where A_H is harvest area in km²
- The H-P envelope plots magnitude (x-axis) vs production loss (y-axis)
- Both axes use logarithmic relationships but are displayed differently:
  - X-axis: Shows the log value itself (magnitude)
  - Y-axis: Shows actual values on a log scale

---

## Example: Understanding the Axes

### Example Event: 100,000 hectares disrupted, 5×10^14 kcal lost

**X-coordinate (Magnitude):**
```python
harvest_ha = 100_000  # hectares
harvest_km2 = harvest_ha * 0.01  # 1,000 km²
magnitude = np.log10(harvest_km2)  # log10(1000) = 3.0
```
**Plotted at:** x = 3.0 (on linear scale)

**Y-coordinate (Production Loss):**
```python
production_loss = 5e14  # kcal (actual value)
```
**Plotted at:** y = 5×10^14 (on log scale, appears at 10^14.7)

---

## Visual Representation

```
Y-axis (log scale)          X-axis (linear scale)
Production Loss [kcal]      Magnitude M_D

10^16 |                     7 = 10,000,000 km²
      |                     6 = 1,000,000 km²
10^15 |    ●                5 = 100,000 km²
      |   /|\               4 = 10,000 km²
10^14 |  / | \              3 = 1,000 km²
      | /  |  \             2 = 100 km²
10^13 |/   |   \
      |    |    \
10^12 |    |     \
      |    |      \
10^11 |    |       \
      |    |        \
10^10 |____|_________\______
      2    3    4    5    6    7
           Magnitude M_D
```

---

## Verification in Code

### Envelope Data Preparation
```python
# From hp_envelope.py lines 56-60
disrupted_areas = np.array(envelope_data['disrupted_areas'])  # km²
upper_bound = np.array(envelope_data['upper_bound'])          # kcal
lower_bound = np.array(envelope_data['lower_bound'])          # kcal

# Calculate magnitude for x-axis
magnitude = np.log10(disrupted_areas)  # Convert km² to magnitude
```

### Events Data Preparation
```python
# From hp_envelope.py line 292
events_data['magnitude'] = np.log10(events_data['harvest_area_km2'])

# From hp_envelope.py line 316
ax.scatter(phase_data['magnitude'], phase_data['production_loss_kcal'])
#          ^^^^^^^^^^^^^^^^^        ^^^^^^^^^^^^^^^^^^^^^^^^^^^
#          X: magnitude (2-7)       Y: actual kcal (10^10-10^16)
```

---

## Common Confusion Points

### ❌ Incorrect Interpretation:
"We're plotting magnitude (2-7) but should plot 10^2 to 10^7"

### ✅ Correct Understanding:
- **X-axis:** We plot magnitude values (2, 3, 4, 5, 6, 7) which represent log₁₀(km²)
- **Y-axis:** We plot actual kcal values (10^10 to 10^16) on a log scale

### Why This Makes Sense:
- Magnitude is a **logarithmic measure** (like Richter scale for earthquakes)
- We plot the magnitude value itself, not 10^magnitude
- The y-axis uses a log scale to show the wide range of production losses
- This creates a meaningful relationship between disrupted area (via magnitude) and production loss

---

## Comparison with Richter Scale

The AgriRichter magnitude works like the Richter scale for earthquakes:

| Richter Magnitude | Energy (Joules) | AgriRichter Magnitude | Harvest Area (km²) |
|-------------------|-----------------|----------------------|-------------------|
| 2.0 | 10^9 J | 2.0 | 100 km² |
| 3.0 | 10^11 J | 3.0 | 1,000 km² |
| 4.0 | 10^13 J | 4.0 | 10,000 km² |
| 5.0 | 10^15 J | 5.0 | 100,000 km² |

When plotting earthquakes, we plot:
- X-axis: Magnitude (2, 3, 4, 5) - not 10^2, 10^3, etc.
- Y-axis: Energy in Joules on log scale

Similarly, for AgriRichter:
- X-axis: Magnitude (2, 3, 4, 5, 6, 7)
- Y-axis: Production loss in kcal on log scale

---

## Conclusion

✅ **The current implementation is CORRECT**

The H-P envelope correctly plots:
- **X-axis:** Magnitude values (2-7) representing log₁₀(harvest area in km²)
- **Y-axis:** Actual production loss values (10^10 to 10^16 kcal) on logarithmic scale

This matches the AgriRichter specification and is analogous to how earthquake magnitude scales are plotted.

**No changes needed to the visualization code.**
