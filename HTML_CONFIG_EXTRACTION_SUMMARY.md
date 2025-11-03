# HTML Configuration Extraction Enhancement

## Problem
The user wanted to enhance the Adobe Launch analyzer to extract Report Suites, Site Domains, and Site Names from HTML configuration patterns that are commonly found in Adobe Launch properties. Most properties have rules that load HTML code with `pfConfig` objects containing these configuration settings.

## Solution Implemented

### 1. New Extraction Functions Added

#### `extract_report_suites_from_config(action_settings)`
- **Purpose**: Extract Report Suites from pfConfig HTML configuration patterns
- **Patterns Supported**:
  - pfConfig `reportSuites` objects with dev/prod environments
  - Traditional Adobe Analytics patterns (`s.account`, `setAccount`, etc.)
  - Report suite patterns in comments and descriptions
- **Output Format**: "Dev: suite1, Prod: suite2" or single suite name

#### `extract_site_domains_from_config(action_settings)`
- **Purpose**: Extract Site Domains from pfConfig HTML configuration patterns
- **Patterns Supported**:
  - pfConfig `domains` objects with dev/prod environments
  - Traditional domain patterns (`s.server`, `trackingServer`, etc.)
  - Domain patterns in comments and descriptions
- **Output Format**: "Dev: domain1, Prod: domain2" or single domain name

#### `extract_site_names_from_config(action_settings)`
- **Purpose**: Extract Site Names from pfConfig HTML configuration patterns
- **Patterns Supported**:
  - pfConfig `siteNames` objects with dev/prod environments
  - All existing eVar61 patterns (backward compatibility)
  - Traditional site name patterns (`site_name`, `setSiteName`, etc.)
  - Context-based patterns in comments
- **Output Format**: "Dev: name1, Prod: name2" or single site name

### 2. Enhanced Main Extraction Function

Updated `extract_cja_tech_strategy_data()` to:
- Use new HTML configuration extraction functions as primary source
- Fall back to traditional extraction methods (eVar61, server domain) if HTML config not found
- Include both new and legacy columns for comparison:
  - `Report Suites` (new)
  - `Site Domains` (new - enhanced)
  - `Site Names` (new - enhanced)
  - `Site Name (eVar 61)` (legacy - kept for comparison)
  - `URL (Server Domain)` (legacy - kept for comparison)

### 3. Enhanced Summary Statistics

Updated `save_cja_tech_strategy_to_csv()` to report:
- Properties with Report Suites found
- Properties with Site Domains found
- Properties with Site Names found
- Legacy statistics maintained for comparison

## Example HTML Configuration Supported

```javascript
var pfConfig = {
  /* Setting Report Suites */
  reportSuites: {
    dev: 'pfizerglobalimpatientsdev',
    prod: 'pfizerglobalimpatientsprod'
  },
  /* Setting Site Domains  */
  domains: {
    dev: 'everydayheroes.co.th',
    prod: 'everydayheroes.co.th'
  },
  /* Setting Site Names */
  siteNames: {
    dev: 'TH PCC Dev Everydayheroes',
    prod: 'TH PCC Prod Everydayheroes'
  },
  // ... other config
};
```

## Test Results

✅ **All Tests Pass**
- Report Suites: Correctly extracts "Dev: pfizerglobalimpatientsdev, Prod: pfizerglobalimpatientsprod"
- Site Domains: Correctly extracts "Dev: everydayheroes.co.th, Prod: everydayheroes.co.th"
- Site Names: Correctly extracts "Dev: TH PCC Dev Everydayheroes, Prod: TH PCC Prod Everydayheroes"

## Key Features

1. **Environment-Aware Extraction**: Handles dev/prod environment pairs
2. **Backward Compatibility**: Maintains existing eVar61 and server domain extraction
3. **Flexible Pattern Matching**: Supports various coding styles and formats
4. **Fallback Logic**: Uses traditional methods if HTML config not found
5. **Comprehensive Coverage**: Handles pfConfig, traditional Adobe Analytics, and comment-based patterns

## Usage

The enhanced analyzer will now automatically:
1. Look for HTML configuration patterns first (most accurate)
2. Fall back to traditional extraction methods if needed
3. Provide both new and legacy columns for comparison
4. Generate comprehensive summary statistics

## Files Modified

- `analyze_simplified.py`: Enhanced with new extraction functions and updated main logic
- `test_config_simple.py`: Created comprehensive test suite
- `HTML_CONFIG_EXTRACTION_SUMMARY.md`: This documentation file

## Benefits

1. **More Accurate Data**: HTML configurations are typically more reliable than scattered JavaScript patterns
2. **Environment Awareness**: Can distinguish between dev and prod configurations
3. **Better Coverage**: Handles the most common Adobe Launch configuration pattern
4. **Future-Proof**: Easy to extend with additional configuration patterns
5. **Backward Compatible**: Existing analysis continues to work unchanged
