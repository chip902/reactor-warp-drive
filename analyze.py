"""This module analyzes data from Adobe Launch and figures out what AdTech is installed on it."""
import re
import csv
from collections import Counter
import pandas as pd
from nltk.tokenize import RegexpTokenizer
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt1
import requests
from urllib.parse import urlparse
import json

# Define your tracking pixels
tracking_pixels = {
    "Facebook Pixel": [
        "fbevents.js",
        "facebook.com/tr",
        "fbpixel.com"
    ],
    "Google Analytics": [
        "analytics.js",
        "google-analytics.com",
        "statcounter.com"
    ],
    "Google Ads": [
        "doubleclick.net",
        "adservice.google"
    ],
    "Hotjar": [
        "hotjar.com/hotjar.js",
        "cdn.hotjar.com/hotjar.js"
    ],
    "Twitter Pixel": [
        "ads-twitter.com",
        "twittershareability.org",
        "twitter.com/i/ads/tracking"
    ],
    "LinkedIn Pixel": [
        "linkedin.com/px",
        "linkedin.com/tr",
        "linkedin-insights.com"
    ],
    "Epsilon": [
        "epsilon.net",
        "consentbox.io"
    ],
    "Hard Coded Adobe Analytics": [
        "adobe.com/analytics",
        "omniture.com"
    ],
    "Microsoft Clarity": [
        "microsoft.com/clarity",
        "clarity.microsoft.com"
    ],
    "Qualtrics": [
        "qualtrics.com"
    ],
    "Oracle Maxymiser": [
        "oracle.com/maxymiser",
        "maxymiser.net",
        "maxymiser.com"
    ],
    "Segment.io": [
        "segment.io"
    ],
    "Mixpanel": [
        "mixpanel.com",
        "mpcdn.net"
    ],
    "Amplitude": [
        "amplitude.com",
        "www.amplitude.com"
    ],
    "Pendo": [
        "pendo.io"
    ],
    "Wix Analytics": [
        "wix.com/ analytics"
    ],
    "ClickMeter": [
        "clickmeter.net"
    ],
    "Crazy Egg": [
        "crazyegg.com",
        "insights.crazyegg.com"
    ],
    "Kissmetrics": [
        "kissmetrics.io",
        "kissmetrics.com"
    ],
    "Heap": [
        "heap.io",
        "www.heap.io"
    ],
    "ClickFunnels": [
        "clickfunnels.com",
        "clickfunnelstracking.com"
    ],
    "Squarespace Analytics": [
        "squarespace.com/ analytics"
    ],
    "Shopify Insights": [
        "shopify.com/insights"
    ],
    "Mailchimp Tracking": [
        "mailchimp.com/tracking",
        "mailchimp.net/tracking"
    ],
    "HubSpot Marketing": [
        "hubspot.com/marketing",
        "hs-analytics.net"
    ],
    "Salesforce DMP": [
        "salesforce.com/dmp",
        "salesforceanalytics.net"
    ],
    "Tapad Pixel": [
        "tapad.com/pixel"
    ],
    "Rubicon Project": [
        "rubiconproject.com/tracking"
    ],
    "DataXu Platform": [
        "dataxup.com/platform"
    ],
    "Sizmek": [
        "sizmek.com",
        "sizmek.net"
    ],
    "Quantcast": [
        "quantcast.com",
        "cdn.quantcast.com"
    ],
    "Chartbeat": [
        "chartbeat.com"
    ],
    "Piwik Analytics": [
        "piwik.org"
    ],
    "Matomo Analytics": [
        "matomo.org"
    ],
    "Ahrefs Tracking": [
        "ahrefs.com/tracking",
        "cdn.ahrefs.com/tracking"
    ],
    "SEMrush Tracking": [
        "semrush.com/tracking",
        "cdn.semrush.com/tracking"
    ],
    "Moz Tracking": [
        "moz.com/tracking",
        "cdn.moz.com/tracking"
    ],
    "Ahrefs Analytics": [
        "ahrefs.com/analytics",
        "cdn.ahrefs.com/analytics"
    ],
    "SEMrush Analytics": [
        "semrush.com/analytics",
        "cdn.semrush.com/analytics"
    ],
    "Moz Analytics": [
        "moz.com/analytics",
        "cdn.moz.com/analytics"
    ],
    "Buffer Tracking": [
        "buffer.com/tracking",
        "cdn.buffer.com/tracking"
    ],
    "Buffer Analytics": [
        "buffer.com/analytics",
        "cdn.buffer.com/analytics"
    ],
    "Google Tag Manager": [
        "googletagmanager.com"
    ],
    "Google Tag Manager 360": [
        "googletagmanager.com/360"
    ],
    "Facebook Custom Audiences": [
        "facebook.com/custom_audiences"
    ],
    "Facebook Pixel with Conversions API": [
        "facebook.com/pixel/conversion-api"
    ],
    "Twitter Website Tags": [
        "twitter.com/website-tags"
    ],
    "LinkedIn Insight Tag": [
        "linkedin.com/insight-tag"
    ],
    "Adobe Experience Cloud": [
        "adobe.com/experience-cloud"
    ],
    "Microsoft Clarity with AI": [
        "microsoft.com/clarity-with-ai"
    ],
    "Qualtrics XM": [
        "qualtrics.com/xm"
    ],
    "Segment.io with AWS Lambda": [
        "segment.io/aws-lambda"
    ],
    "Mixpanel with Firebase Analytics": [
        "mixpanel.com/firebase-analytics"
    ],
    "Amplitude with Snowflake": [
        "amplitude.com/snowflake"
    ],
    "Pendo with Salesforce": [
        "pendo.io/salesforce"
    ],
    "Wix Analytics with Google Cloud": [
        "wix.com/google-cloud"
    ],
    "ClickMeter with A/B Testing": [
        "clickmeter.net/ab-testing"
    ],
    "Crazy Egg with User Feedback": [
        "crazyegg.com/user-feedback"
    ],
    "Kissmetrics with Machine Learning": [
        "kissmetrics.io/machine-learning"
    ],
    "Heap with JavaScript": [
        "heap.io/javascript"
    ],
    "ClickFunnels with Sales Funnels": [
        "clickfunnels.com/sales-funnels"
    ],
    "Squarespace Analytics with SEO": [
        "squarespace.com/seo"
    ],
    "Shopify Insights with Customer Journey": [
        "shopify.com/customer-journey"
    ],
    "Mailchimp Tracking with Email Marketing": [
        "mailchimp.com/email-marketing"
    ],
    "HubSpot Marketing with CRM": [
        "hubspot.com/crm"
    ],
    "Salesforce DMP with Advertising": [
        "salesforce.com/advertising"
    ],
    "Tapad Pixel with Mobile App Tracking": [
        "tapad.com/mobile-app-tracking"
    ],
    "Rubicon Project with Native Ads": [
        "rubiconproject.com/native-ads"
    ],
    "DataXu Platform with Programmatic Advertising": [
        "dataxup.com/programmatic-advertising"
    ],
    "AdRoll": [
        "a.roll-1.com",
        "adroll.com/tracking"
    ],
    "Criteo": [
        "criteo.net",
        "criteo.com/tracking"
    ],
    "The Trade Desk (TTD)": [
        "thetradedesk.com/tracking"
    ],
    "AppNexus": [
        "appnexus.net",
        "appnexus.com/tracking"
    ],
    "Yahoo Dot Pixels": [
        "b.yahoo.com",
        "dot-pixel.yahoo.co.jp"
    ],
    "Bing Ads": [
        "bat.bing.com",
        "adnexus.net/bingads"
    ],
    "Pinterest Pixel": [
        "pinterest.com/px",
        "ct.pinterest.com"
    ],
    "Snapchat Pixel": [
        "snapchat.com/pixel",
        "sc-static.net/pixel"
    ],
    "TikTok Ads Pixel": [
        "tiktok.com/ads/pixel",
        "analytics.tiktok.com"
    ],
    "Yandex Metrica": [
        "mc.yandex.ru",
        "metrika.yandex.com"
    ],
    "VKontakte Pixel": [
        "vk.com/pixel",
        "vkontakte.ru/pixel"
    ],
    "Yahoo Flurry Analytics": [
        "flurry.com/analytics",
        "data.flurry.com/analytics"
    ],
    "Adobe Advertising Cloud (formerly Tubemogul)": [
        "adobedtm.com/tubemogul",
        "adobedc.net/tubemogul"
    ],
    "Salesforce Marketing Cloud (ExactTarget)": [
        "salesforce.com/marketingcloud",
        "exacttarget.com/tracking"
    ],
    "Optimizely": [
        "optimizely.net",
        "optimizely.com/tracking"
    ],
    "FullStory": [
        "fullstory.com/tracking"
    ],
    "Heap with React Native": [
        "heap.io/react-native"
    ],
    "Sentry for Error Tracking": [
        "sentry.io/error-tracking",
        "sentry-cdn.com/error-tracking"
    ],
    "Baidu Tongji": [
        "hm.baidu.com",
        "tongji.baidu.com"
    ],
    "Alibaba Analytics": [
        "alibaba.com/analytics",
        "aliyuncs.com/analytics"
    ],
    "Yahoo Gemini": [
        "gemini.yahoo.com",
        "admanager.yahoo.com/gemini"
    ],
    "Verizon Media (Oath)": [
        "verizondigitalmedia.com",
        "oath.com/tracking"
    ],
    "AOL Advertising": [
        "aolads.com",
        "adtech.aol.com"
    ],
    "Index Exchange (IX)": [
        "indexexchange.com",
        "ix-dsp.com"
    ],
    "AppsFlyer": [
        "appsflyer.net",
        "attribution-service.com"
    ],
    "Adjust": [
        "adjust.com/tracking",
        "adjust.io/tracking"
    ],
    "Branch Metrics": [
        "branch.io/metrics",
        "bnc.lt/metrics"
    ],
    "Kochava": [
        "kochava.net",
        "kochavatrk.com"
    ],
    "Tune": [
        "tune.com/tracking",
        "mobile-service.com"
    ],
    "RadiumOne": [
        "radiumone.com/tracking"
    ],
    "Rocket Fuel": [
        "rfihub.net",
        "rocketfuel.com/tracking"
    ],
    "Adform": [
        "adform.net",
        "adform.com/tracking"
    ],
    "The Nielsen Company (Nielsen Digital Ad Ratings)": [
        "nielsen-online.com/dart",
        "nielsen.com/digital-ad-ratings"
    ],
    "ComScore": [
        "scorecardresearch.com",
        "comscore.net"
    ],
    "IAS (Integral Ad Science)": [
        "integralads.com",
        "iasds001.com"
    ],
    "DoubleVerify": [
        "doubleverify.com/tracking",
        "dvtag.net/tracking"
    ],
    "Integral Ad Science (IAS)": [
        "integralads.com",
        "iasds001.com"
    ],
    "MediaMath": [
        "mediamath.com",
        "mm-adnet.com"
    ],
    "The Rubicon Project (Rubicon)": [
        "rubiconproject.com",
        "fastflip.com/rubicon"
    ],
    "Adserver Plus": [
        "adserverplus.com",
        "as-us.com"
    ],
    "Smaato": [
        "smaato.net",
        "smaato.com/tracking"
    ],
    "InMobi": [
        "inmobi.com",
        "w.inmobi.com"
    ],
    "Unity Ads": [
        "unityads.unity3d.com",
        "unityads.com"
    ],
    "IronSource": [
        "ironsrc.net",
        "iron-src.com"
    ],
    "Vungle": [
        "vungle.com/tracking",
        "static.vungle.com/tracking"
    ],
    "Chartboost": [
        "chartboost.com",
        "answerscloud.com/chartboost"
    ],
    "Applovin": [
        "applovin.com/tracking",
        "applvn.com/tracking"
    ],
    "StartApp": [
        "startapp.com/tracking",
        "startappexchange.com/tracking"
    ],
    "Tapsense": [
        "tapsense.net",
        "tapsense-analytics.com"
    ],
    "Tapjoy": [
        "tapjoy.com/tracking",
        "tapjoyads.com/tracking"
    ],
    "Supersonic Ads": [
        "supersonicads.net",
        "supersonicads-server.com"
    ],
    "Nanigans (now part of Adobe)": [
        "nanigans.net",
        "adobedc.net/nanigans"
    ],
    "Celtra": [
        "celtra.com/tracking",
        "celtratech.net/tracking"
    ]
}


def extract_js_function_calls(texts):
    # Use regular expression to find all function calls in the Action Settings column
    js_functions = []
    for text in texts:
        # Find JavaScript function call names with 3 or more characters
        matches = re.findall(r'\b\w{3,}\b', text)

        # Check if "function" appears and extract the next word as a separate function name
        words = re.split(r'\s+', text)
        for i in range(len(words) - 1):
            if words[i].lower() == 'function' and len(words[i+1]) >= 3:
                matches.append(re.escape(words[i+1]))

        js_functions.extend(matches)

    # Count the frequency of each function call name
    freq = Counter(js_functions)

    return freq


def save_js_functions_to_csv(js_functions):
    # Convert dictionary to DataFrame
    df_js_functions = pd.DataFrame(list(js_functions.items()), columns=[
                                   'JavaScript Function', 'Count'])
    # Save DataFrame to CSV with quoting to handle embedded newlines and commas
    df_js_functions.to_csv("significant_js_functions.csv",
                           index=False, quoting=csv.QUOTE_ALL)
    print("Significant JavaScript functions saved to significant_js_functions.csv")


def save_tracking_pixels_to_csv(results):
    # Convert list of results to DataFrame
    flattened_results = []
    for result in results:
        for source in result["Source Code"]:
            flattened_results.append({
                "Property Name": result["Property Name"],
                "Detected Pixel": source["pixel_name"],
                "Source Code": source["source_code"]
            })
    df_pixels = pd.DataFrame(flattened_results)
    # Save DataFrame to CSV with quoting to handle embedded newlines and commas
    df_pixels.to_csv("tracking_pixel_report_with_source.csv",
                     index=False, quoting=csv.QUOTE_ALL)
    print("Tracking pixel report saved to tracking_pixel_report_with_source.csv")


def find_tracking_pixels(action_text, pixel_dict):
    detected_pixels = Counter()
    detected_source = []

    for pixel_name, pixel_patterns in pixel_dict.items():
        for pattern in pixel_patterns:
            if pattern in action_text:
                detected_pixels[pixel_name] += 1
                detected_source.append(
                    {"pixel_name": pixel_name, "source_code": action_text})
    return detected_pixels, detected_source


def extract_significant_functions(texts, initial_js_dict, pixel_dict, property_names):
    tokenizer = RegexpTokenizer(r'\w+')
    js_token_counts = Counter(initial_js_dict)
    pixel_counts = Counter()
    results = []

    for i, text in enumerate(tqdm(texts, desc="Processing text data")):
        property_name = property_names[i]

        # Tokenize and count JS functions
        tokens = tokenizer.tokenize(text.lower())
        js_token_counts.update(tokens)

        # Check for tracking pixels in the same text
        detected_pixels, detected_source = find_tracking_pixels(
            text, pixel_dict)

        # Add result to array if there are detected pixels
        if detected_pixels:
            results.append({
                "Property Name": property_name,
                "Detected Pixels": detected_pixels,
                "Source Code": detected_source
            })

        # Update pixel_counts with the results of find_tracking_pixels (whether or not pixels were detected)
        pixel_counts += detected_pixels

    # Filter significant JS functions
    significant_js_functions = {token: count for token,
                                count in js_token_counts.items() if count > 1}
    return significant_js_functions, pixel_counts, results


def visualize_tracking_pixels(pixel_data):
    df = pd.DataFrame(list(pixel_data.items()), columns=[
                      'Tracking Pixel', 'Count'])
    # Sort the values by 'Count' in descending order
    df = df.sort_values(by="Count", ascending=False)
    # Plot the data
    plt1.figure(figsize=(10, 8))
    plt1.barh(df['Tracking Pixel'], df['Count'], color='skyblue')
    plt1.xlabel('Count')
    plt1.ylabel('Tracking Pixel')
    plt1.title('3rd-Party Tracking Pixels Detected')
    plt1.gca().invert_yaxis()  # Invert the Y-axis to have the largest on top
    plt1.show()


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


def extract_site_name_from_evar61(action_settings):
    """
    Extract site name from eVar61 patterns in action settings
    """
    evar61_patterns = [
        # Original eVar61 patterns
        r'evar61["\']?\s*[:=]\s*["\']([^"\']+)["\']',
        r's\.eVar61["\']?\s*[:=]\s*["\']([^"\']+)["\']',
        r'["\']eVar61["\']?\s*[:=]\s*["\']([^"\']+)["\']',
        r'variables\.eVar61["\']?\s*[:=]\s*["\']([^"\']+)["\']',
        r'setEVar\(61,\s*["\']([^"\']+)["\']',
        r's\.eVar61\s*=\s*["\']([^"\']+)["\']',
        r'61["\']?\s*[:=]\s*["\']([^"\']+)["\']',

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

    for pattern in evar61_patterns:
        matches = re.findall(pattern, text_to_analyze, re.IGNORECASE)
        if matches:
            # Return the first non-empty match, cleaned up
            for match in matches:
                if match.strip() and len(match.strip()) > 2:
                    return match.strip()

    return 'Not Found'


def extract_server_domain_from_actions(action_settings, property_domains=None, primary_domain=None):
    """
    Extract server domain from Adobe Launch property configuration
    Now uses property-level domains from the API
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

    # Priority 4: Window/document location (main site domain)
    location_patterns = [
        r'window\.location\.hostname\s*[:=]\s*["\']([^"\']+)["\']',
        r'document\.location\.hostname\s*[:=]\s*["\']([^"\']+)["\']',
        r'location\.host\s*[:=]\s*["\']([^"\']+)["\']',
        r'window\.location\.host\s*[:=]\s*["\']([^"\']+)["\']',
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

    # Try Priority 4: Location-based patterns
    for pattern in location_patterns:
        matches = re.findall(pattern, text_to_analyze, re.IGNORECASE)
        for match in matches:
            cleaned = clean_domain(match)
            if cleaned and not is_third_party_domain(cleaned):
                return cleaned

    # Fallback: Look for domains that don't match third-party patterns
    all_domains = re.findall(
        r'\b([a-zA-Z0-9-]+(?:\.[a-zA-Z0-9-]+)+)\b', text_to_analyze)
    for domain in all_domains:
        cleaned = clean_domain(domain)
        if cleaned and not is_third_party_domain(cleaned) and len(domain) > 8:
            return cleaned

    return 'Not Found'


def extract_cja_tech_strategy_data(df):
    """
    Extract CJA/WebSDK Tech Strategy data for each property
    Now uses enhanced property-level data from app.py
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

        # Extract required data
        site_name = extract_site_name_from_evar61(action_settings)
        server_domain = extract_server_domain_from_actions(
            action_settings, property_domains, primary_domain)
        cms = detect_cms_from_action_settings(action_settings)

        # Count Launch rules (number of rows for this property)
        launch_rules_count = len(group)

        # Use property-level site name if eVar61 not found and property domains exist
        if site_name == 'Not Found' and primary_domain and primary_domain != 'Not Found':
            site_name = primary_domain.replace('www.', '')

        results.append({
            'Site Name (eVar 61)': site_name,
            'URL (Server Domain)': server_domain,
            'Web Property': property_name,
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
        f"Properties with eVar61 found: {sum(1 for r in results if r['Site Name (eVar 61)'] != 'Not Found')}")
    print(
        f"Properties with server domain found: {sum(1 for r in results if r['URL (Server Domain)'] != 'Not Found')}")

    return df_cja


def run_cja_analysis_only():
    """
    Standalone function to run only CJA/WebSDK Tech Strategy analysis
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


def main():
    # Starting dictionary of JavaScript functions/keywords
    starting_dictionary = {
        "fbevents.js": 0,
        "gtag": 0,
        "Hotjar": 0,
        "ctrk": 0,
        "yimg": 0,
        "epsilon": 0
    }

    try:
        # Load the CSV file
        df = pd.read_csv(
            "adobe_launch_rules_with_actions_filtered.csv", low_memory=False)

        # Ensure the relevant columns exist
        if "Action Settings" not in df.columns or "Property Name" not in df.columns:
            raise Exception(
                "The CSV file must contain 'Action Settings' and 'Property Name' columns.")

        # Extract action settings and property names for NLP analysis
        action_settings = df["Action Settings"].dropna().str.lower().tolist()
        property_names = df["Property Name"].dropna().tolist()
        js_function_calls_freq = extract_js_function_calls(action_settings)

        # Perform NLP and tracking pixel detection
        significant_js_functions, pixel_counts, results = extract_significant_functions(
            action_settings, starting_dictionary, tracking_pixels, property_names
        )

        # Save the significant JS functions to a CSV file
        save_js_functions_to_csv(significant_js_functions)

        # Save the tracking pixel counts and raw source data to a CSV file
        save_tracking_pixels_to_csv(results)

        print("Significant JavaScript functions and tracking pixel analysis complete.")

        # NEW: Extract and save CJA/WebSDK Tech Strategy data
        print("\n" + "="*50)
        print("Starting CJA/WebSDK Tech Strategy Analysis...")
        print("="*50)

        cja_results = extract_cja_tech_strategy_data(df)
        cja_df = save_cja_tech_strategy_to_csv(cja_results)

        print("CJA/WebSDK Tech Strategy analysis complete.")
        print("="*50)

        # Visualize the tracking pixel data
        visualize_tracking_pixels(pixel_counts)

        # Visualize the top 5 most frequent JavaScript function call names
        top_10_js_functions = sorted(
            js_function_calls_freq.items(), key=lambda x: x[1], reverse=True)[:10]

        js_functions, freqs = zip(*top_10_js_functions)
        plt.bar(js_functions, freqs)
        plt.xlabel('JS Function Call')
        plt.ylabel('Frequency')
        plt.title('Top 10 Most Frequent JS Function Calls')
        plt.show()

    except Exception as e:
        print(f"Error during processing: {e}")


if __name__ == "__main__":
    import nltk
    import sys

    # Download required NLTK data
    nltk.download('stopwords')
    nltk.download('punkt')

    # Check command line arguments
    if len(sys.argv) > 1 and sys.argv[1] == "--cja-only":
        # Run only CJA/WebSDK Tech Strategy analysis
        run_cja_analysis_only()
    else:
        # Run full analysis (original behavior)
        main()

    print("\n" + "="*50)
    print("Analysis Options:")
    print("- Run full analysis: python analyze.py")
    print("- Run CJA analysis only: python analyze.py --cja-only")
    print("="*50)
