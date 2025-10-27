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
