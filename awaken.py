from encodings import utf_8
from sklearn.feature_extraction.text import TfidfVectorizer
from matplotlib import pyplot as plt
import pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
from collections import defaultdict
import json
from tqdm import tqdm


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
        "doubleclick",
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
        "consentbox.io"  # used by Epsilon for cookie consent
    ],
    "Adobe Analytics": [
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
    ]
}


# Function to compute TF-IDF and find important terms


# Function to compute TF-IDF and find important terms


def extract_significant_terms(texts):
    # TF-IDF vectorizer
    vectorizer = TfidfVectorizer(stop_words='english')
    X = vectorizer.fit_transform(texts)

    # Sum the TF-IDF values for each word
    terms = vectorizer.get_feature_names_out()
    scores = X.sum(axis=0).A1
    term_scores = {terms[i]: scores[i] for i in range(len(terms))}

    # Sort by importance (descending)
    sorted_terms = dict(
        sorted(term_scores.items(), key=lambda item: item[1], reverse=True))

    return sorted_terms


def main():
    try:
        # Load the CSV file
        df = pd.read_csv("adobe_launch_rules_with_actions.csv")

        # Ensure the relevant columns exist
        if "Action Settings" not in df.columns:
            raise Exception(
                "The CSV file must contain 'Action Settings' column.")

        # Extract text data for NLP analysis
        action_settings = df["Action Settings"].dropna().tolist()

        # Perform AI-driven analysis using TF-IDF
        significant_terms = extract_significant_terms(action_settings)
        print("Significant terms found:", significant_terms)

        # Save significant terms to a JSON file
        with open("significant_js_terms.json", "w", encoding='utf-8') as f:
            json.dump(significant_terms, f, indent=4)

    except Exception as e:
        print("Error during processing:", e)


if __name__ == "__main__":
    main()
