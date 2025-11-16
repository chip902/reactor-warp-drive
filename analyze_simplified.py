"""Simplified Adobe Launch analyzer focused on CJA/WebSDK Tech Strategy data extraction."""

import re
import csv
import pandas as pd
from tqdm import tqdm


def detect_cms_from_action_settings(action_settings):
    """
    Detect Content Management System based on action settings and JavaScript patterns
    """
    cms_patterns = {
        'WordPress': ['wp-content', 'wp-includes', 'wp-json', 'wordpress', 'wp-admin'],
        'Drupal': ['drupal', 'sites/default', 'modules/node', 'themes/bartik'],
        'Joomla': ['joomla', 'com_content', 'components/com_', 'templates/ja_'],
        'Magento': ['magento', 'Mage.php', 'skin/frontend', 'js/mage/'],
        'Shopify': ['shopify.com', 'cdn.shopify.com', 'Shopify.theme', 'Shopify.shop'],
        'Wix': ['wix.com', 'wixstatic.com', 'wix-code-bridge', 'wix-cloud'],
        'Squarespace': ['squarespace.com', 'static.squarespace.com', 'sqs-block'],
        'Sitecore': ['sitecore', '/sitecore/', 'sc_emailform'],
        'Adobe Experience Manager': ['aem', '/content/dam/', '/etc.clientlibs/'],
        'HubSpot CMS': ['hubspot.com/cms', 'hs-content', 'hubspotcms'],
        'Contentful': ['contentful.com', 'cdn.contentful.com', 'contentful'],
        'Strapi': ['strapi.io', '/api/', 'strapi'],
        'Ghost': ['ghost.org', 'ghost-api', 'ghost-content'],
        'Craft CMS': ['craftcms.com', 'craft.js', 'craft/'],
        'TYPO3': ['typo3', 'typo3conf', 'typo3temp'],
        'Concrete5': ['concrete5', 'concrete-', 'ccm_'],
        'ExpressionEngine': ['expressionengine', 'ee/', 'exp:'],
        'SilverStripe': ['silverstripe', 'ss-', 'framework/'],
        'Umbraco': ['umbraco', '/umbraco/', 'umbraco-']
    }

    detected_cms = []
    text_to_analyze = ' '.join(action_settings).lower()

    for cms, patterns in cms_patterns.items():
        for pattern in patterns:
            if pattern in text_to_analyze:
                detected_cms.append(cms)
                break

    return ', '.join(detected_cms) if detected_cms else 'Not Detected'


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

    # Helper to detect domain-like strings (e.g., site.com, www.example.co.uk)
    def looks_like_domain(value: str) -> bool:
        value = value.strip()
        # Basic domain pattern: something.suffix (optionally with subdomains)
        return bool(re.search(r"\b[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}\b", value))

    found_domain_like = False

    for pattern in report_suite_patterns:
        matches = re.findall(pattern, text_to_analyze,
                             re.IGNORECASE | re.DOTALL)
        if matches:
            # Handle tuple matches from dev/prod patterns
            if isinstance(matches[0], tuple):
                dev_suite, prod_suite = matches[0]

                prod_clean = prod_suite.strip() if prod_suite else ""
                dev_clean = dev_suite.strip() if dev_suite else ""

                # Prefer non-domain prod, then non-domain dev
                if prod_clean and not looks_like_domain(prod_clean):
                    return f"Dev: {dev_clean}, Prod: {prod_clean}" if dev_clean else prod_clean
                if dev_clean and not looks_like_domain(dev_clean):
                    return f"Dev: {dev_clean}"

                # Both look like domains; remember and continue searching
                if prod_clean or dev_clean:
                    if (prod_clean and looks_like_domain(prod_clean)) or (dev_clean and looks_like_domain(dev_clean)):
                        found_domain_like = True
            else:
                # Handle single matches
                for match in matches:
                    cleaned = match.strip()
                    if not cleaned or len(cleaned) <= 2:
                        continue
                    if looks_like_domain(cleaned):
                        found_domain_like = True
                        continue
                    return cleaned

    # If we only found domain-like values, treat as blank/none rather than "Not Found"
    if found_domain_like:
        return ""

    return 'Not Found'


def extract_site_domains_from_config(action_settings):
    """
    Extract Site Domains from pfConfig HTML configuration patterns
    IMPORTANT: This extracts actual domains (e.g., 'www.pfizerpro.com')
    NOT Report Suite IDs (RSIDs like 'pfizerglobalimpatientsprod')
    """
    site_domain_patterns = [
        # pfConfig domains patterns - Handle both : and = after domains, with comments and newlines
        r'domains\s*[=:]\s*\{[^}]*?dev\s*:\s*["\'"]+([^"\'<]+?)["\'"]+[^}]*?prod\s*:\s*["\'"]+([^"\'<]+?)["\'"]+',
        r'domains\s*[=:]\s*\{[^}]*?prod\s*:\s*["\'"]+([^"\'<]+?)["\'"]+[^}]*?dev\s*:\s*["\'"]+([^"\'<]+?)["\'"]+',

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

    # Patterns to EXCLUDE (these are RSIDs, not domains)
    rsid_patterns = [
        # e.g., pfizerglobalimpatientsprod
        r'^[a-z]+global[a-z]+(dev|prod|development|production)$',
        # Generic RSID pattern
        r'^[a-z]+[a-z0-9]+(dev|prod|development|production|staging)$',
    ]

    # Placeholder patterns to exclude
    placeholder_patterns = [
        r'<<.*>>',  # Template placeholders
        r'<.*>',    # Generic placeholders
    ]

    text_to_analyze = ' '.join(action_settings)

    for pattern in site_domain_patterns:
        matches = re.findall(pattern, text_to_analyze,
                             re.IGNORECASE | re.DOTALL)
        if matches:
            # Handle tuple matches from dev/prod patterns
            if isinstance(matches[0], tuple):
                dev_domain, prod_domain = matches[0]

                # Validate that these are NOT RSIDs or placeholders
                is_prod_rsid = any(re.match(rsid_pat, prod_domain.strip(
                ), re.IGNORECASE) for rsid_pat in rsid_patterns)
                is_prod_placeholder = any(re.search(
                    ph_pat, prod_domain.strip(), re.IGNORECASE) for ph_pat in placeholder_patterns)

                # ONLY return prod value (we don't care about dev)
                if not is_prod_rsid and not is_prod_placeholder and prod_domain and prod_domain.strip():
                    return prod_domain.strip()
            else:
                # Handle single matches - validate not an RSID or placeholder
                for match in matches:
                    is_rsid = any(re.match(rsid_pat, match.strip(), re.IGNORECASE)
                                  for rsid_pat in rsid_patterns)
                    is_placeholder = any(re.search(ph_pat, match.strip(
                    ), re.IGNORECASE) for ph_pat in placeholder_patterns)
                    if not is_rsid and not is_placeholder and match.strip() and len(match.strip()) > 2:
                        return match.strip()

    return 'Not Found'


def extract_site_names_from_config(action_settings):
    """
    Extract Site Names from pfConfig HTML configuration patterns
    IMPORTANT: This extracts the friendly site name (e.g., 'TH PCC Prod Everydayheroes')
    NOT the Report Suite ID (RSID). This should match eVar61 values.
    """
    site_name_patterns = [
        # pfConfig siteNames patterns - MOST SPECIFIC FIRST
        # Handle both : and = after siteNames, with comments and newlines
        r'siteNames\s*[=:]\s*\{[^}]*?dev\s*:\s*["\'"]+([^"\'<]+?)["\'"]+[^}]*?prod\s*:\s*["\'"]+([^"\'<]+?)["\'"]+',
        r'siteNames\s*[=:]\s*\{[^}]*?prod\s*:\s*["\'"]+([^"\'<]+?)["\'"]+[^}]*?dev\s*:\s*["\'"]+([^"\'<]+?)["\'"]+',

        # Traditional site name patterns (existing eVar61 patterns)
        r'evar61["\']?\s*[:=]\s*["\']([^"\']+)["\']',
        r's\.eVar61["\']?\s*[:=]\s*["\']([^"\']+)["\']',
        r'["\']eVar61["\']?\s*[:=]\s*["\']([^"\']+)["\']',
        r'variables\.eVar61["\']?\s*[:=]\s*["\']([^"\']+)["\']',
        r'setEVar\(61,\s*["\']([^"\']+)["\']',
        r's\.eVar61\s*=\s*["\']([^"\']+)["\']',

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

        # Context-based patterns (might appear in comments or descriptions)
        r'"site name"[^:]*:\s*["\']([^"\']+)["\']',
        r'"siteName"[^:]*:\s*["\']([^"\']+)["\']',
        r'Site Name[:\s]*["\']([^"\']+)["\']',
        r'SITE_NAME[:\s]*["\']([^"\']+)["\']'
    ]

    # Patterns to EXCLUDE (these are RSIDs, not site names, or placeholders)
    rsid_patterns = [
        # e.g., pfizerglobalimpatientsprod
        r'^[a-z]+global[a-z]+(dev|prod|development|production)$',
        # Generic RSID pattern
        r'^[a-z]+[a-z0-9]+(dev|prod|development|production|staging)$',
    ]

    # Placeholder patterns to exclude
    placeholder_patterns = [
        r'<<.*>>',  # Template placeholders like <<US PCC Dev Asset-Name>>
        r'<.*>',    # Generic placeholders
    ]

    text_to_analyze = ' '.join(action_settings)

    for pattern in site_name_patterns:
        matches = re.findall(pattern, text_to_analyze,
                             re.IGNORECASE | re.DOTALL)
        if matches:
            # Handle tuple matches from dev/prod patterns
            if isinstance(matches[0], tuple):
                dev_name, prod_name = matches[0]

                # Validate that these are NOT RSIDs or placeholders
                is_prod_rsid = any(re.match(rsid_pat, prod_name.strip(
                ), re.IGNORECASE) for rsid_pat in rsid_patterns)
                is_prod_placeholder = any(re.search(
                    ph_pat, prod_name.strip(), re.IGNORECASE) for ph_pat in placeholder_patterns)

                # ONLY return prod value (we don't care about dev)
                if not is_prod_rsid and not is_prod_placeholder and prod_name and prod_name.strip():
                    return prod_name.strip()
            else:
                # Handle single matches - validate not an RSID or placeholder
                for match in matches:
                    is_rsid = any(re.match(rsid_pat, match.strip(), re.IGNORECASE)
                                  for rsid_pat in rsid_patterns)
                    is_placeholder = any(re.search(ph_pat, match.strip(
                    ), re.IGNORECASE) for ph_pat in placeholder_patterns)
                    if not is_rsid and not is_placeholder and match.strip() and len(match.strip()) > 2:
                        return match.strip()

    return 'Not Found'


def extract_site_name_from_evar61(action_settings):
    """
    Extract site name from eVar61 patterns and site name variations in action settings
    IMPORTANT: This should extract the EXACT SAME value as extract_site_names_from_config()
    Uses the same patterns to ensure consistency between Site Names and eVar61 columns
    """
    # USE THE EXACT SAME PATTERNS AS extract_site_names_from_config()
    site_name_patterns = [
        # pfConfig siteNames patterns - MOST SPECIFIC FIRST
        # Handle both : and = after siteNames, with comments and newlines
        r'siteNames\s*[=:]\s*\{[^}]*?dev\s*:\s*["\'"]+([^"\'<]+?)["\'"]+[^}]*?prod\s*:\s*["\'"]+([^"\'<]+?)["\'"]+',
        r'siteNames\s*[=:]\s*\{[^}]*?prod\s*:\s*["\'"]+([^"\'<]+?)["\'"]+[^}]*?dev\s*:\s*["\'"]+([^"\'<]+?)["\'"]+',

        # Traditional site name patterns (existing eVar61 patterns)
        r'evar61["\']?\s*[:=]\s*["\']([^"\']+)["\']',
        r's\.eVar61["\']?\s*[:=]\s*["\']([^"\']+)["\']',
        r'["\']eVar61["\']?\s*[:=]\s*["\']([^"\']+)["\']',
        r'variables\.eVar61["\']?\s*[:=]\s*["\']([^"\']+)["\']',
        r'setEVar\(61,\s*["\']([^"\']+)["\']',
        r's\.eVar61\s*=\s*["\']([^"\']+)["\']',

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

        # Context-based patterns (might appear in comments or descriptions)
        r'"site name"[^:]*:\s*["\']([^"\']+)["\']',
        r'"siteName"[^:]*:\s*["\']([^"\']+)["\']',
        r'Site Name[:\s]*["\']([^"\']+)["\']',
        r'SITE_NAME[:\s]*["\']([^"\']+)["\']'
    ]

    # Patterns to EXCLUDE (these are RSIDs, not site names) - SAME AS extract_site_names_from_config
    rsid_patterns = [
        # e.g., pfizerglobalimpatientsprod
        r'^[a-z]+global[a-z]+(dev|prod|development|production)$',
        # Generic RSID pattern
        r'^[a-z]+[a-z0-9]+(dev|prod|development|production|staging)$',
    ]

    # Placeholder patterns to exclude - SAME AS extract_site_names_from_config
    placeholder_patterns = [
        r'<<.*>>',  # Template placeholders like <<US PCC Dev Asset-Name>>
        r'<.*>',    # Generic placeholders
    ]

    text_to_analyze = ' '.join(action_settings)

    for pattern in site_name_patterns:
        matches = re.findall(pattern, text_to_analyze,
                             re.IGNORECASE | re.DOTALL)
        if matches:
            # Handle tuple matches from dev/prod patterns
            if isinstance(matches[0], tuple):
                dev_name, prod_name = matches[0]

                # Validate that these are NOT RSIDs or placeholders
                is_prod_rsid = any(re.match(rsid_pat, prod_name.strip(
                ), re.IGNORECASE) for rsid_pat in rsid_patterns)
                is_prod_placeholder = any(re.search(
                    ph_pat, prod_name.strip(), re.IGNORECASE) for ph_pat in placeholder_patterns)

                # ONLY return prod value (we don't care about dev) - SAME AS extract_site_names_from_config
                if not is_prod_rsid and not is_prod_placeholder and prod_name and prod_name.strip():
                    return prod_name.strip()
            else:
                # Handle single matches - validate not an RSID or placeholder
                for match in matches:
                    is_rsid = any(re.match(rsid_pat, match.strip(), re.IGNORECASE)
                                  for rsid_pat in rsid_patterns)
                    is_placeholder = any(re.search(ph_pat, match.strip(
                    ), re.IGNORECASE) for ph_pat in placeholder_patterns)
                    if not is_rsid and not is_placeholder and match.strip() and len(match.strip()) > 2:
                        return match.strip()

    return 'Not Found'


def extract_server_domain_from_actions(action_settings, property_domains=None, primary_domain=None):
    """
    Extract server domain from Adobe Launch property configuration
    Uses property-level domains from API when available
    """
    # Priority 1: Use property-level domains from API (most accurate)
    if primary_domain and primary_domain != "Not Found":
        return primary_domain

    # Priority 2: Use any property-level domains
    if property_domains and property_domains != "Not Found":
        domains = property_domains.split(";")
        if domains and domains[0].strip():
            return domains[0].strip()

    # Priority 3: Adobe Analytics/Launch specific server configurations
    launch_server_patterns = [
        r's\.server\s*=\s*["\']([^"\']+)["\']',
        r'["\']server["\']?\s*[:=]\s*["\']([^"\']+)["\']',
        r'variables\.server["\']?\s*[:=]\s*["\']([^"\']+)["\']',
        r'setServer\(["\']([^"\']+)["\']',
        r'trackingServer["\']?\s*[:=]\s*["\']([^"\']+)["\']',
        r'["\']trackingServer["\']?\s*[:=]\s*["\']([^"\']+)["\']',
        r'measurement["\']?\s*[:=]\s*["\']([^"\']+)["\']',
        r'["\']measurement["\']?\s*[:=]\s*["\']([^"\']+)["\']',
    ]

    # Exclude known third-party domains that aren't the main site
    excluded_domains = [
        'adobedtm.com', 'omtrdc.net', 'sc.omtrdc.net', 'metrics.adobedtm.com',
        'google-analytics.com', 'googletagmanager.com', 'doubleclick.net',
        'facebook.com', 'facebook.net', 'connect.facebook.net',
        'linkedin.com', 'licdn.com', 'analytics.linkedin.com',
        'twitter.com', 'twimg.com', 'analytics.twitter.com',
        'yahoo.com', 'yimg.com', 'analytics.yahoo.com',
        'hotjar.com', 'cdn.hotjar.com',
        'quantserve.com', 'quantcount.com',
        'scorecardresearch.com',
        'taboola.com', 'trc.taboola.com',
        'outbrain.com', 'amplitude.com',
        'segment.io', 'mixpanel.com',
        'optimizely.com', 'cdn.optimizely.com',
        'onestag.com', 'tealium.com', 'tiqcdn.com',
        'cookielaw.org', 'onetrust.com', 'cdn.cookielaw.org',
        'cookiebot.com', 'consentmanager.net',
        'adnxs.com', 'adnxs.net', 'criteo.com',
        'rubiconproject.com', 'indexexchange.com',
        'amazon-adsystem.com', 'c.amazon-adsystem.com',
        'googleadservices.com', 'google.com',
        'googlesyndication.com', 'googleads.g.doubleclick.net'
    ]

    def is_third_party_domain(domain):
        """Check if domain is a known third-party/analytics domain"""
        domain_lower = domain.lower()
        for excluded in excluded_domains:
            if excluded.lower() in domain_lower:
                return True
        return False

    def clean_domain(domain):
        """Clean and validate domain"""
        domain = domain.strip()
        # Remove protocols
        domain = re.sub(r'https?://', '', domain)
        # Remove paths and query strings
        domain = re.sub(r'/.*$', '', domain)
        # Remove port numbers
        domain = re.sub(r':\d+$', '', domain)
        # Basic domain validation
        if re.match(r'^[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$', domain):
            return domain
        return None

    text_to_analyze = ' '.join(action_settings)

    # Try Priority 3: Adobe-specific server configurations
    for pattern in launch_server_patterns:
        matches = re.findall(pattern, text_to_analyze, re.IGNORECASE)
        for match in matches:
            cleaned = clean_domain(match)
            if cleaned and not is_third_party_domain(cleaned):
                return cleaned

    return 'Not Found'


def extract_cja_tech_strategy_data(df):
    """
    Extract CJA/WebSDK Tech Strategy data for each property
    Uses enhanced property-level data from app.py and HTML configuration patterns
    """
    results = []

    # Check if we have the enhanced columns from updated app.py
    has_property_data = all(col in df.columns for col in [
                            'Property Domains', 'Primary Domain', 'Property Platform'])

    # Group by property to get all actions for each property
    grouped = df.groupby('Property Name')

    for property_name, group in tqdm(grouped, desc="Extracting CJA Tech Strategy data"):
        # Get all action settings for this property
        action_settings = group['Action Settings'].dropna().tolist()

        # Extract property-level data if available
        property_domains = group['Property Domains'].iloc[0] if has_property_data else None
        primary_domain = group['Primary Domain'].iloc[0] if has_property_data else None
        platform = group['Property Platform'].iloc[0] if has_property_data else None

        # Extract configuration data from HTML patterns
        report_suites = extract_report_suites_from_config(action_settings)
        site_domains = extract_site_domains_from_config(action_settings)
        site_names = extract_site_names_from_config(action_settings)

        # Extract traditional data for fallback
        site_name_evar61 = extract_site_name_from_evar61(action_settings)
        server_domain = extract_server_domain_from_actions(
            action_settings, property_domains, primary_domain)
        cms = detect_cms_from_action_settings(action_settings)

        # Count Launch rules (number of rows for this property)
        launch_rules_count = len(group)

        # Use HTML config site names first, fallback to eVar61, then property domains
        final_site_name = site_names
        if final_site_name == 'Not Found' and site_name_evar61 != 'Not Found':
            final_site_name = site_name_evar61
        elif final_site_name == 'Not Found' and primary_domain and primary_domain != 'Not Found':
            final_site_name = primary_domain.replace('www.', '')

        # Use HTML config domains first, fallback to server domain extraction
        final_server_domain = site_domains
        if final_server_domain == 'Not Found':
            final_server_domain = server_domain

        # Split Report Suites into Dev/Prod columns when possible
        rsid_dev = ''
        rsid_prod = ''
        if isinstance(report_suites, str):
            # Look for explicit Dev/Prod labels
            dev_match = re.search(
                r"Dev:\s*([^,]+)", report_suites, re.IGNORECASE)
            prod_match = re.search(
                r"Prod:\s*([^,]+)", report_suites, re.IGNORECASE)
            if dev_match:
                rsid_dev = dev_match.group(1).strip()
            if prod_match:
                rsid_prod = prod_match.group(1).strip()
            # If no labeled values but we have a single RSID string, prefer Prod
            if not rsid_dev and not rsid_prod:
                single = report_suites.strip()
                if single and single.lower() != 'not found':
                    rsid_prod = single

        results.append({
            'Web Property': property_name,
            'Report Suites': report_suites,
            'RSID (Dev)': rsid_dev,
            'RSID (Prod)': rsid_prod,
            'Site Domains': final_server_domain,
            'Site Names': final_site_name,
            'Site Name (eVar 61)': site_name_evar61,  # Keep for comparison
            'URL (Server Domain)': server_domain,    # Keep for comparison
            'Content Management System': cms,
            'Number of Launch Rules': launch_rules_count,
            'Platform': platform if platform else 'Unknown'
        })

    return results


def save_cja_tech_strategy_to_csv(results):
    """
    Save CJA/WebSDK Tech Strategy data to CSV
    """
    df_cja = pd.DataFrame(results)

    # Save DataFrame to CSV
    df_cja.to_csv("cja_websdk_tech_strategy.csv",
                  index=False, quoting=csv.QUOTE_ALL)
    print("CJA/WebSDK Tech Strategy data saved to cja_websdk_tech_strategy.csv")

    # Also save a summary
    print(f"\nSummary: {len(results)} properties analyzed")
    print(
        f"Properties with detected CMS: {sum(1 for r in results if r['Content Management System'] != 'Not Detected')}")
    print(
        f"Properties with RSIDs found: {sum(1 for r in results if (r.get('RSID (Dev)', '').strip() or r.get('RSID (Prod)', '').strip()))}")
    print(
        f"Properties with Site Domains found: {sum(1 for r in results if r['Site Domains'] != 'Not Found')}")
    print(
        f"Properties with Site Names found: {sum(1 for r in results if r['Site Names'] != 'Not Found')}")
    print(
        f"Properties with eVar61 found: {sum(1 for r in results if r['Site Name (eVar 61)'] != 'Not Found')}")
    print(
        f"Properties with server domain found: {sum(1 for r in results if r['URL (Server Domain)'] != 'Not Found')}")

    return df_cja


def main():
    """
    Main function for CJA/WebSDK Tech Strategy analysis
    """
    try:
        # Load the CSV file
        df = pd.read_csv(
            "adobe_launch_rules_with_actions_filtered.csv", low_memory=False)

        # Ensure the relevant columns exist
        if "Action Settings" not in df.columns or "Property Name" not in df.columns:
            raise Exception(
                "The CSV file must contain 'Action Settings' and 'Property Name' columns.")

        print("Starting CJA/WebSDK Tech Strategy Analysis...")
        print("="*50)

        cja_results = extract_cja_tech_strategy_data(df)
        cja_df = save_cja_tech_strategy_to_csv(cja_results)

        print("CJA/WebSDK Tech Strategy analysis complete.")
        print("="*50)

        # Display first few rows as preview
        print("\nPreview of results:")
        print(cja_df.head().to_string(index=False))

        return cja_df

    except Exception as e:
        print(f"Error during CJA analysis: {e}")
        return None


if __name__ == "__main__":
    main()
    print("\n" + "="*50)
    print("CJA/WebSDK Tech Strategy Analysis Complete")
    print("="*50)
