# Multi-Tier Envelope System: User Guide

## 🎯 **Overview**

The Multi-Tier Envelope System provides policy-relevant agricultural capacity analysis by filtering agricultural land based on productivity levels. This guide helps you select the appropriate tier for your analysis and interpret the results effectively.

## 📊 **Understanding Tiers**

### **Comprehensive Tier (All Lands)**
- **Description:** Includes all agricultural land, including marginal areas
- **Yield Range:** 0-100th percentile
- **Data Retention:** ~100% of original cells
- **Use Cases:**
  - Academic research and theoretical bounds
  - Baseline comparisons
  - Maximum theoretical capacity assessment
- **Target Users:** Researchers, academics, theoretical analysis

### **Commercial Tier (Economically Viable)**
- **Description:** Excludes bottom 20% of yields (marginal/subsistence farming)
- **Yield Range:** 20-100th percentile  
- **Data Retention:** ~80% of original cells
- **Width Reduction:** 25-35% narrower bounds than comprehensive
- **Use Cases:**
  - Government planning and policy development
  - Investment decisions and market analysis
  - Food security planning
  - Trade capacity assessment
- **Target Users:** Policy makers, government agencies, investors, planners

## 🎯 **Tier Selection Guidelines**

### **Choose Comprehensive Tier When:**
- Conducting academic research on theoretical limits
- Establishing baseline comparisons
- Analyzing maximum possible production under ideal conditions
- Publishing scientific papers requiring complete data coverage

### **Choose Commercial Tier When:**
- Developing government food security policies
- Planning agricultural investments
- Assessing realistic trade capacity
- Analyzing economically viable production scenarios
- Creating policy recommendations for decision makers

## 🚀 **Quick Start Guide**

### **Basic Analysis**

```python
from agririchter.analysis import MultiTierEnvelopeEngine
from agririchter.data import SPAMDataLoader

# Load your crop data
loader = SPAMDataLoader()
wheat_data = loader.load_crop_data('wheat')

# Initialize multi-tier engine
engine = MultiTierEnvelopeEngine()

# For policy analysis (recommended)
policy_results = engine.calculate_single_tier(wheat_data, tier='commercial')

# For academic research
research_results = engine.calculate_single_tier(wheat_data, tier='comprehensive')

# Compare both tiers
comparison = engine.calculate_all_tiers(wheat_data)
```

### **National Analysis**

```python
from agririchter.analysis import NationalEnvelopeAnalyzer

# Analyze specific country
analyzer = NationalEnvelopeAnalyzer('USA')
usa_results = analyzer.analyze_national_capacity(wheat_data, tier='commercial')

# Generate policy report
policy_report = analyzer.create_food_security_assessment(usa_results)
```

### **Country Comparison**

```python
from agririchter.analysis import NationalComparisonAnalyzer

# Compare multiple countries
comparator = NationalComparisonAnalyzer()
countries = ['USA', 'CHN', 'BRA']
comparison_report = comparator.compare_countries(countries, 'wheat', tier='commercial')
```

## 📈 **Interpreting Results**

### **Envelope Width Reductions**
- **25-35% reduction:** Typical for commercial tier vs comprehensive
- **Higher reductions:** Indicate more heterogeneous agricultural productivity
- **Lower reductions:** Suggest more uniform agricultural systems

### **Data Retention Rates**
- **>80%:** Good retention, tier filtering effective
- **60-80%:** Moderate retention, some productivity stratification
- **<60%:** High filtering, significant marginal agriculture excluded

### **Policy Implications**

#### **Commercial Tier Results Indicate:**
- **Realistic production capacity** for economic planning
- **Investment-worthy agricultural areas** for development
- **Trade-relevant production** for export/import analysis
- **Food security baseline** for policy planning

#### **Width Reduction Benefits:**
- **More precise capacity estimates** for planning
- **Reduced uncertainty** in policy scenarios
- **Better risk assessment** for food security
- **Improved investment targeting** for development

## 🌍 **Country-Specific Considerations**

### **United States**
- **Recommended Tier:** Commercial (focuses on efficient agriculture)
- **Key Regions:** Corn Belt, Great Plains, California
- **Policy Focus:** Export capacity, climate resilience
- **Typical Width Reduction:** 30-35%

### **China**
- **Recommended Tier:** Comprehensive or Commercial (food security focus)
- **Key Regions:** Northeast, North China Plain, Yangtze River
- **Policy Focus:** Self-sufficiency, urbanization pressure
- **Typical Width Reduction:** 25-30%

### **Developing Countries**
- **Recommended Tier:** Comprehensive (includes subsistence farming)
- **Policy Focus:** Food security, rural development
- **Considerations:** Higher proportion of marginal agriculture

## ⚠️ **Common Pitfalls and Best Practices**

### **Avoid These Mistakes:**
1. **Using comprehensive tier for policy planning** - Includes unrealistic marginal areas
2. **Using commercial tier for academic completeness** - Excludes important subsistence systems
3. **Ignoring data retention rates** - Low retention may indicate insufficient data
4. **Comparing different tiers directly** - Always use same tier for comparisons

### **Best Practices:**
1. **Start with commercial tier** for most policy applications
2. **Document tier selection rationale** in reports
3. **Report both width reduction and retention rates**
4. **Validate results against known agricultural patterns**
5. **Use consistent tiers** across comparative analyses

## 🔧 **Advanced Usage**

### **Custom Tier Configuration**

```python
from agririchter.analysis import TierConfiguration

# Create custom tier (e.g., high-productivity agriculture)
custom_tier = TierConfiguration(
    name='High Productivity',
    description='Top 50% of yields only',
    yield_percentile_min=50,
    yield_percentile_max=100,
    policy_applications=['intensive_agriculture', 'export_focus'],
    target_users=['agribusiness', 'export_planners']
)

# Use custom tier
results = engine.calculate_single_tier(crop_data, custom_tier=custom_tier)
```

### **Regional Analysis**

```python
# Analyze specific regions within countries
regional_analyzer = RegionalEnvelopeAnalyzer('USA')
corn_belt_results = regional_analyzer.analyze_region('corn_belt', 'maize', tier='commercial')
```

### **Scenario Analysis**

```python
# Run policy scenarios with different tiers
scenario_analyzer = PolicyScenarioAnalyzer()
scenarios = ['drought_resilience', 'trade_disruption', 'climate_adaptation']
scenario_results = scenario_analyzer.run_scenarios(base_results, scenarios)
```

## 📊 **Output Formats and Visualization**

### **Standard Outputs**
- **Envelope bounds arrays:** Upper and lower production limits
- **Width reduction metrics:** Quantified improvement over comprehensive
- **Data quality reports:** Retention rates and filtering statistics
- **Validation results:** Mathematical property verification

### **Visualization Options**
- **Envelope comparison plots:** Show tier differences
- **Geographic maps:** Spatial patterns of filtered data
- **Width reduction charts:** Quantify improvements
- **Policy scenario plots:** Compare different analyses

## 🆘 **Troubleshooting**

### **Low Width Reductions (<10%)**
- **Cause:** Uniform agricultural productivity in dataset
- **Solution:** Check data quality, consider different crops/regions

### **High Data Loss (>50%)**
- **Cause:** Aggressive filtering or poor data quality
- **Solution:** Validate SPAM filtering, check yield distributions

### **Unrealistic Results**
- **Cause:** Incorrect tier selection or data issues
- **Solution:** Validate against known agricultural statistics

### **Performance Issues**
- **Cause:** Large datasets or inefficient processing
- **Solution:** Use caching, parallel processing, or data subsampling

## 📚 **Additional Resources**

- **Technical Documentation:** See `MULTI_TIER_TECHNICAL_GUIDE.md`
- **Policy Guide:** See `MULTI_TIER_POLICY_GUIDE.md`
- **API Reference:** See `MULTI_TIER_API_REFERENCE.md`
- **Mathematical Methodology:** See `docs/ENVELOPE_BUILDER_MATHEMATICAL_GUIDE.md`

## 🤝 **Support and Community**

For questions, issues, or contributions:
- **Technical Issues:** Check troubleshooting section above
- **Policy Questions:** Consult policy guide and use cases
- **Development:** See technical documentation for system architecture

---

**Remember: Choose commercial tier for policy applications, comprehensive tier for academic research. Always document your tier selection rationale and validate results against known agricultural patterns.**