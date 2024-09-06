from matplotlib import pyplot as plt
import pandas as pd
from collections import Counter
from nltk.tokenize import RegexpTokenizer
from tqdm import tqdm
import json

# Define your tracking pixels
tracking_pixels = {
    "Facebook Pixel": [
        "fbevents.js",
        "facebook.com/tr",
        "fbpixel.com"
    ],
    "Google Analytics": [
        "gtag.js",
        "analytics.js",
        "google-analytics.com",
        "statcounter.com"  # not exactly a marketing pixel, but sometimes used
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
# Function to check if a script contains tracking pixels


def find_tracking_pixels(action_text, pixel_dict):
    detected_pixels = Counter()
    for pixel_name, pixel_patterns in pixel_dict.items():
        for pattern in pixel_patterns:
            if pattern in action_text:
                detected_pixels[pixel_name] += 1
    return detected_pixels

# Function to perform NLP on text data for JavaScript functions and tracking pixels


def extract_significant_functions(texts, initial_js_dict, pixel_dict):
    tokenizer = RegexpTokenizer(r'\w+')
    js_token_counts = Counter(initial_js_dict)
    pixel_counts = Counter()

    for text in tqdm(texts, desc="Processing text data"):
        # Tokenize and count JS functions
        tokens = tokenizer.tokenize(text.lower())
        js_token_counts.update(tokens)

        # Check for tracking pixels in the same text
        pixel_counts.update(find_tracking_pixels(text, pixel_dict))

    # Filter significant JS functions
    significant_js_functions = {token: count for token,
                                count in js_token_counts.items() if count > 1}
    return significant_js_functions, pixel_counts

# Function to visualize the tracking pixel data


def visualize_tracking_pixels(pixel_data):
    df = pd.DataFrame(pixel_data.items(), columns=[
                      "Tracking Pixel", "Count"]).sort_values(by="Count", ascending=False)

    plt.figure(figsize=(10, 8))
    plt.barh(df["Tracking Pixel"], df["Count"], color="skyblue")
    plt.xlabel("Count")
    plt.ylabel("Tracking Pixel")
    plt.title("3rd-Party Tracking Pixels Detected")
    plt.gca().invert_yaxis()
    plt.show()

# Main analysis function


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
        df = pd.read_csv("adobe_launch_rules_with_actions.csv")

        # Ensure the relevant columns exist
        if "Action Settings" not in df.columns:
            raise Exception(
                "The CSV file must contain 'Action Settings' column.")

        # Extract action settings for NLP analysis
        action_settings = df["Action Settings"].dropna().str.lower().tolist()

        # Perform NLP and tracking pixel detection
        significant_js_functions, pixel_counts = extract_significant_functions(
            action_settings, starting_dictionary, tracking_pixels)

        # Save the significant JS functions to a JSON file
        with open("significant_js_functions.json", "w") as f:
            json.dump(significant_js_functions, f, indent=4)

        # Save the tracking pixel counts to a JSON file
        with open("tracking_pixel_report.json", "w") as f:
            json.dump(pixel_counts, f, indent=4)

        print("Significant JavaScript functions and tracking pixel analysis complete.")

        # Visualize the tracking pixel data
        visualize_tracking_pixels(pixel_counts)

    except Exception as e:
        print(f"Error during processing: {e}")


if __name__ == "__main__":
    import nltk
    nltk.download('stopwords')
    nltk.download('punkt')
    main()
