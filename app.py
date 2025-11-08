#!/usr/bin/env python3
"""Test script to demonstrate the new HTML configuration extraction functions."""

from analyze_simplified import (
    extract_report_suites_from_config,
    extract_site_domains_from_config,
    extract_site_names_from_config
)
import sys
import os

# Add the current directory to Python path to import our module
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


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


if __name__ == "__main__":
    test_extraction_functions()
