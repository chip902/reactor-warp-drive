#!/usr/bin/env python3
"""Simple test script to demonstrate the new HTML configuration extraction functions."""

import re

def extract_report_suites_from_config(action_settings):
    """
    Extract Report Suites from pfConfig HTML configuration patterns
    """
    report_suite_patterns = [
        # pfConfig reportSuites patterns
        r'reportSuites\s*:\s*\{\s*dev\s*:\s*["\']([^"\']+)["\']\s*,\s*prod\s*:\s*["\']([^"\']+)["\']',
        r'reportSuites\s*:\s*\{\s*prod\s*:\s*["\']([^"\']+)["\']\s*,\s*dev\s*:\s*["\']([^"\']+)["\']',
        r'dev\s*:\s*["\']([^"\']+)["\']\s*,\s*prod\s*:\s*["\']([^"\']+)["\']',
        r'prod\s*:\s*["\']([^"\']+)["\']\s*,\s*dev\s*:\s*["\']([^"\']+)["\']',
        
        # Traditional Adobe Analytics patterns
        r's\.account\s*=\s*["\']([^"\']+)["\']',
        r'["\']account["\']?\s*[:=]\s*["\']([^"\']+)["\']',
        r'variables\.account["\']?\s*[:=]\s*["\']([^"\']+)["\']',
        r'setAccount\(["\']([^"\']+)["\']',
        r'trackingAccount["\']?\s*[:=]\s*["\']([^"\']+)["\']',
        r's_account\s*=\s*["\']([^"\']+)["\']',
        r's\.su\s*=\s*["\']([^"\']+)["\']',
        r'["\']s\.account["\']?\s*[:=]\s*["\']([^"\']+)["\']',
        
        # Report suite patterns in comments or descriptions
        r'report\s*suite[^:]*:\s*["\']([^"\']+)["\']',
        r'reportSuite[^:]*:\s*["\']([^"\']+)["\']',
        r'suiteid[^:]*:\s*["\']([^"\']+)["\']',
        r'rsid[^:]*:\s*["\']([^"\']+)["\']'
    ]

    text_to_analyze = ' '.join(action_settings)

    for pattern in report_suite_patterns:
        matches = re.findall(pattern, text_to_analyze, re.IGNORECASE | re.DOTALL)
        if matches:
            # Handle tuple matches from dev/prod patterns
            if isinstance(matches[0], tuple):
                dev_suite, prod_suite = matches[0]
                if prod_suite and prod_suite.strip():
                    return f"Dev: {dev_suite.strip()}, Prod: {prod_suite.strip()}"
                elif dev_suite and dev_suite.strip():
                    return f"Dev: {dev_suite.strip()}"
            else:
                # Handle single matches
                for match in matches:
                    if match.strip() and len(match.strip()) > 2:
                        return match.strip()

    return 'Not Found'


def extract_site_domains_from_config(action_settings):
    """
    Extract Site Domains from pfConfig HTML configuration patterns
    """
    site_domain_patterns = [
        # pfConfig domains patterns
        r'domains\s*:\s*\{\s*dev\s*:\s*["\']([^"\']+)["\']\s*,\s*prod\s*:\s*["\']([^"\']+)["\']',
        r'domains\s*:\s*\{\s*prod\s*:\s*["\']([^"\']+)["\']\s*,\s*dev\s*:\s*["\']([^"\']+)["\']',
        r'dev\s*:\s*["\']([^"\']+)["\']\s*,\s*prod\s*:\s*["\']([^"\']+)["\']',
        r'prod\s*:\s*["\']([^"\']+)["\']\s*,\s*dev\s*:\s*["\']([^"\']+)["\']',
        
        # Traditional domain patterns
        r's\.server\s*=\s*["\']([^"\']+)["\']',
        r'["\']server["\']?\s*[:=]\s*["\']([^"\']+)["\']',
        r'variables\.server["\']?\s*[:=]\s*["\']([^"\']+)["\']',
        r'setServer\(["\']([^"\']+)["\']',
        r'trackingServer["\']?\s*[:=]\s*["\']([^"\']+)["\']',
        r'["\']trackingServer["\']?\s*[:=]\s*["\']([^"\']+)["\']',
        r'measurement["\']?\s*[:=]\s*["\']([^"\']+)["\']',
        r'["\']measurement["\']?\s*[:=]\s*["\']([^"\']+)["\']',
        
        # Domain patterns in comments or descriptions
        r'domain[^:]*:\s*["\']([^"\']+)["\']',
        r'site\s*domain[^:]*:\s*["\']([^"\']+)["\']',
        r'host[^:]*:\s*["\']([^"\']+)["\']'
    ]

    text_to_analyze = ' '.join(action_settings)

    for pattern in site_domain_patterns:
        matches = re.findall(pattern, text_to_analyze, re.IGNORECASE | re.DOTALL)
        if matches:
            # Handle tuple matches from dev/prod patterns
            if isinstance(matches[0], tuple):
                dev_domain, prod_domain = matches[0]
                if prod_domain and prod_domain.strip():
                    return f"Dev: {dev_domain.strip()}, Prod: {prod_domain.strip()}"
                elif dev_domain and dev_domain.strip():
                    return f"Dev: {dev_domain.strip()}"
            else:
                # Handle single matches
                for match in matches:
                    if match.strip() and len(match.strip()) > 2:
                        return match.strip()

    return 'Not Found'


def extract_site_names_from_config(action_settings):
    """
    Extract Site Names from pfConfig HTML configuration patterns
    """
    site_name_patterns = [
        # pfConfig siteNames patterns
        r'siteNames\s*:\s*\{\s*dev\s*:\s*["\']([^"\']+)["\']\s*,\s*prod\s*:\s*["\']([^"\']+)["\']',
        r'siteNames\s*:\s*\{\s*prod\s*:\s*["\']([^"\']+)["\']\s*,\s*dev\s*:\s*["\']([^"\']+)["\']',
        r'dev\s*:\s*["\']([^"\']+)["\']\s*,\s*prod\s*:\s*["\']([^"\']+)["\']',
        r'prod\s*:\s*["\']([^"\']+)["\']\s*,\s*dev\s*:\s*["\']([^"\']+)["\']',
        
        # Traditional site name patterns (existing eVar61 patterns)
        r'evar61["\']?\s*[:=]\s*["\']([^"\']+)["\']',
        r's\.eVar61["\']?\s*[:=]\s*["\']([^"\']+)["\']',
        r'["\']evar61["\']?\s*[:=]\s*["\']([^"\']+)["\']',
        r'variables\.evar61["\']?\s*[:=]\s*["\']([^"\']+)["\']',
        r'setEVar\(61,\s*["\']([^"\']+)["\']',
        r's\.eVar61\s*=\s*["\']([^"\']+)["\']',
        r'61["\']?\s*[:=]\s*["\']([^"\']+)["\"]',
        
        # Site Name variations
        r'site_name["\']?\s*[:=]\s*["\']([^"\']+)["\']',
        r'["\']site_name["\']?\s*[:=]\s*["\']([^"\']+)["\']',
        r'siteName["\']?\s*[:=]\s*["\']([^"\']+)["\']',
        r'["\']siteName["\']?\s*[:=]\s*["\']([^"\']+)["\']',
        r'site\.name["\']?\s*[:=]\s*["\']([^"\']+)["\']',
        r'["\']site\.name["\']?\s*[:=]\s*["\']([^"\']+)["\']',
        r'variables\.site_name["\']?\s*[:=]\s*["\']([^"\']+)["\']',
        r'variables\.siteName["\']?\s*[:=]\s*["\']([^"\']+)["\']',
        r'setSiteName\(["\']([^"\']+)["\']',
        r's\.siteName\s*=\s*["\']([^"\']+)["\']',
        r's\.site_name\s*=\s*["\']([^"\']+)["\']',
        
        # Other common variations
        r'website_name["\']?\s*[:=]\s*["\']([^"\']+)["\']',
        r'["\']website_name["\']?\s*[:=]\s*["\']([^"\']+)["\']',
        r'websiteName["\']?\s*[:=]\s*["\']([^"\']+)["\']',
        r'["\']websiteName["\']?\s*[:=]\s*["\']([^"\']+)["\']',
        r'domain_name["\']?\s*[:=]\s*["\']([^"\']+)["\']',
        r'["\']domain_name["\']?\s*[:=]\s*["\']([^"\']+)["\']',
        r'property_name["\']?\s*[:=]\s*["\']([^"\']+)["\']',
        r'["\']property_name["\']?\s*[:=]\s*["\']([^"\']+)["\']',
        
        # Context-based patterns (might appear in comments or descriptions)
        r'"site name"[^:]*:\s*["\']([^"\']+)["\']',
        r'"siteName"[^:]*:\s*["\']([^"\']+)["\']',
        r'Site Name[:\s]*["\']([^"\']+)["\']',
        r'SITE_NAME[:\s]*["\']([^"\']+)["\']'
    ]

    text_to_analyze = ' '.join(action_settings)

    for pattern in site_name_patterns:
        matches = re.findall(pattern, text_to_analyze, re.IGNORECASE | re.DOTALL)
        if matches:
            # Handle tuple matches from dev/prod patterns
            if isinstance(matches[0], tuple):
                dev_name, prod_name = matches[0]
                if prod_name and prod_name.strip():
                    return f"Dev: {dev_name.strip()}, Prod: {prod_name.strip()}"
                elif dev_name and dev_name.strip():
                    return f"Dev: {dev_name.strip()}"
            else:
                # Handle single matches
                for match in matches:
                    if match.strip() and len(match.strip()) > 2:
                        return match.strip()

    return 'Not Found'


# Sample HTML configuration based on the user's example
sample_config = [
    """
    <script>
      //set default environment mode
      var envMode = 'dev';
      //obtain mode from _satellite object based on the deployed environment
      if (typeof(_satellite.buildInfo.environment) !== 'undefined' && (_satellite.buildInfo.environment !== '')){
        var scriptEnvironment = _satellite.buildInfo.environment;
        //console.log("Launch environment is " + scriptEnvironment);
        var envMode = '';
        if (scriptEnvironment === 'production'){
         envMode = 'prod'; 
        } else if (scriptEnvironment === 'development'){
          envMode = 'dev';
        } else{
          envMode = 'dev';
        }
      }
      var pfConfig = {
      //automatically set to 'dev' when deployed to development or staging, or to 'prod' when deployed to production
      mode: envMode,
      /*****************************************************/    
      /***** Do not change anything above this line ********/
      /*****************************************************/
      /* Report Suites for Dev and Prod
          For PCC: dev: 'pfizerglobalimpatientsdev', prod: 'pfizerglobalimpatientsprod'
          For INT: dev: 'pfizerglobalimintdev', prod: 'pfizerglobalimintprod'
          For HCP: dev: 'pfizerglobalimdevelopment', prod: 'pfizerglobalimprod'
      */
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
      /* Setting Site Section Tracking Strategy */
      siteSection: {
        prefix: '',
        prefixHost: false,
        delimiter: '>'
      },
      /* Setting Page Naming Convension Strategy */
      pageName: {
        base: 'path',
        prefix: '',
        delimiter: '>',
        homePage: 'home'
      },
      /* track IP address, true - yes (default), false - no */
      enableIPAddressTracking: true,
      /* Setting Custom Link Tracking Strategy */
      customLinks: [],
      /* Setting Comma Separated Campaign Parameters */
      campaignParams: 'cmp,cbn,cid',
      /* Setting Tracking Servers for FPC */
      trackingServer: '',
      trackingServerSecure: '',
      /* Setting Country Code */
      siteCountry: 'TH',
      /* Setting Visitor ID Flag (FPC is required) */
      enableVisitorId: true
    };
    </script>
    """
]

def test_extraction_functions():
    """Test the new extraction functions with sample HTML configuration."""
    
    print("Testing HTML Configuration Extraction Functions")
    print("=" * 60)
    
    # Test Report Suites extraction
    print("\n1. Testing Report Suites Extraction:")
    report_suites = extract_report_suites_from_config(sample_config)
    print(f"   Result: {report_suites}")
    
    # Test Site Domains extraction
    print("\n2. Testing Site Domains Extraction:")
    site_domains = extract_site_domains_from_config(sample_config)
    print(f"   Result: {site_domains}")
    
    # Test Site Names extraction
    print("\n3. Testing Site Names Extraction:")
    site_names = extract_site_names_from_config(sample_config)
    print(f"   Result: {site_names}")
    
    print("\n" + "=" * 60)
    print("Extraction test complete!")
    
    # Expected results based on the sample config
    print("\nExpected Results:")
    print("- Report Suites: Dev: pfizerglobalimpatientsdev, Prod: pfizerglobalimpatientsprod")
    print("- Site Domains: Dev: everydayheroes.co.th, Prod: everydayheroes.co.th")
    print("- Site Names: Dev: TH PCC Dev Everydayheroes, Prod: TH PCC Prod Everydayheroes")
    
    # Check if results match expectations
    print("\nValidation:")
    expected_report_suites = "Dev: pfizerglobalimpatientsdev, Prod: pfizerglobalimpatientsprod"
    expected_site_domains = "Dev: everydayheroes.co.th, Prod: everydayheroes.co.th"
    expected_site_names = "Dev: TH PCC Dev Everydayheroes, Prod: TH PCC Prod Everydayheroes"
    
    print(f"✓ Report Suites: {'PASS' if report_suites == expected_report_suites else 'FAIL'}")
    print(f"✓ Site Domains: {'PASS' if site_domains == expected_site_domains else 'FAIL'}")
    print(f"✓ Site Names: {'PASS' if site_names == expected_site_names else 'FAIL'}")

if __name__ == "__main__":
    test_extraction_functions()
