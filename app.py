import os
import json
import requests
import pandas as pd
from tqdm import tqdm
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Configuration
api_key = os.getenv("API_KEY")
client_id = os.getenv("CLIENT_ID")
client_secret = os.getenv("CLIENT_SECRET")
company_id = os.getenv("COMPANY_ID")
org_id = os.getenv("ORG_ID")

# URLs
base_url = "https://reactor.adobe.io"
token_url = "https://ims-na1.adobelogin.com/ims/token/v3"
properties_url = f"{base_url}/companies/{company_id}/properties"

# Headers
headers = {
    "x-api-key": api_key,
    "Accept": "application/vnd.api+json;revision=1",
}


def get_token(client_id, client_secret):
    data = {
        "grant_type": "client_credentials",
        "client_id": client_id,
        "client_secret": client_secret,
        "scope": "AdobeID,openid,read_organizations,additional_info.job_function,additional_info.projectedProductContext,additional_info.roles",
    }
    response = requests.post(token_url, data=data, headers={
                             "Content-Type": "application/x-www-form-urlencoded"})
    if response.status_code != 200:
        raise Exception(f"Failed to get access token: {response.text}")
    return response.json()["access_token"]


def get_all_properties(access_token):
    auth_headers = {
        **headers,
        "Authorization": f"Bearer {access_token}",
        "x-gw-ims-org-id": org_id,
        "Content-Type": "application/vnd.api+json"
    }
    properties = []
    page = 1

    while True:
        response = requests.get(f"{properties_url}?page[number]={
                                page}&page[size]=100", headers=auth_headers)
        if response.status_code != 200:
            raise Exception(f"Failed to fetch properties: {response.text}")

        data = response.json()["data"]
        properties.extend(data)

        if "next" not in response.json().get("links", {}):
            break
        page += 1

    return properties


def get_rules_for_property(property_id, access_token):
    auth_headers = {
        **headers,
        "Authorization": f"Bearer {access_token}",
        "x-gw-ims-org-id": org_id
    }
    rules_url = f"{base_url}/properties/{property_id}/rules"
    response = requests.get(rules_url, headers=auth_headers)
    if response.status_code != 200:
        raise Exception(f"Failed to fetch rules for property {
                        property_id}: {response.text}")
    rules = response.json()["data"]
    return rules


def get_actions_for_rule(property_id, rule_id, access_token):
    auth_headers = {
        **headers,
        "Authorization": f"Bearer {access_token}",
        "x-gw-ims-org-id": org_id
    }
    actions_url = f"{base_url}/rules/{rule_id}/rule_components"
    response = requests.get(actions_url, headers=auth_headers)
    if response.status_code == 404:
        print(f"Actions not found for rule {
              rule_id} in property {property_id}. Skipping.")
        return []
    if response.status_code != 200:
        raise Exception(f"Failed to fetch actions for rule {
                        rule_id}: {response.text}")
    actions = response.json()["data"]
    return actions


def save_progress(all_rules):
    df = pd.DataFrame(all_rules)
    df.to_csv("adobe_launch_rules_with_actions.csv", index=False)


def main():
    try:
        access_token = get_token(client_id, client_secret)
        properties = get_all_properties(access_token)
        print(f"Total properties fetched: {len(properties)}")

        all_rules = []

        for prop in tqdm(properties, desc="Processing properties"):
            prop_id = prop["id"]
            prop_name = prop["attributes"]["name"]

            try:
                rules = get_rules_for_property(prop_id, access_token)
                for rule in tqdm(rules, desc=f"Processing rules for property {prop_id}", leave=False):
                    rule_id = rule["id"]
                    rule_name = rule["attributes"]["name"]
                    rule_description = rule["attributes"].get(
                        "description", "")

                    try:
                        actions = get_actions_for_rule(
                            prop_id, rule_id, access_token)
                        for action in actions:
                            action_data = {
                                "Property ID": prop_id,
                                "Property Name": prop_name,
                                "Rule ID": rule_id,
                                "Rule Name": rule_name,
                                "Rule Description": rule_description,
                                "Action ID": action["id"],
                                "Action Name": action["attributes"]["name"],
                                "Action Settings": action["attributes"]["settings"],
                            }
                            all_rules.append(action_data)
                    except Exception as e:
                        print(f"Failed to fetch actions for rule {
                              rule_id}: {e}")
                save_progress(all_rules)  # Save progress after each property
            except Exception as e:
                print(e)

        print("Data saved to adobe_launch_rules_with_actions.csv")

    except Exception as e:
        print("Error during initialization:", e)


if __name__ == "__main__":
    main()
