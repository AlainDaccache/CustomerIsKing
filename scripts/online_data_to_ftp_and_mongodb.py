"""
In this script, we 
1. Download the data from Google Drive
2. Enrich it with fake customer addresses
3. Split the data: 
        - customer info (incl. addresses) sent to file server (Nginx)
        - transaction info (excl. addresses) sent to OLTP db (MongoDB)
"""

import pandas as pd
import json

import requests
from pymongo import MongoClient
from ucimlrepo import fetch_ucirepo
import os

# For the HR pdf
doc_urls = [
    "https://www.cgi.com/sites/default/files/2021-03/cgi-hr-and-payroll-bps-fact-sheet.pdf",
    "https://www.cgi.com/sites/default/files/2019-01/cgi-cloud-native-solutions.pdf",
    "https://www.cgi.com/sites/default/files/2021-11/customer_relationship_management_solutions.pdf",
    "https://www.cgi.com/sites/default/files/2022-09/cgi-brochure-canada-data_analytics-ai_offerings.pdf",
    "https://www.cgi.com/sites/default/files/2019-08/artificial_intelligence_in_the_making_a_pathway_to_success_final.pdf",
    "https://www.cgi.com/sites/default/files/2022-04/cgi_conversational_ai_practice_brochure.pdf",
    "https://www.cgi.com/sites/default/files/2022-03/cgi_cgi_advanced_analytics_infographic.pdf",
    "https://www.cgi.com/sites/default/files/2021-10/cgi-advantage-intelligence-suite_1.pdf",
]
doc_filenames = [url.split("/")[-1] for url in doc_urls]
print("Doc Filenames:", doc_filenames)
# f
destination_file_path = "customer_transactions.json"

# MongoDB configurations
mongodb_uri = "mongodb://root:example@localhost:27017/"
database_name = "my_ecom_mongodb"
collection_name = "customer-transactions"


def download_file(url, filename):
    # Send a GET request to the URL
    response = requests.get(url)

    # Check if the request was successful (status code 200)
    if response.status_code == 200:
        # Open the file in binary write mode and write the content from the response
        with open(filename, "wb") as file:
            file.write(response.content)
        print("File downloaded successfully!")
    else:
        print(f"Failed to download file. Status code: {response.status_code}")


for url, f in zip(doc_urls, doc_filenames):
    download_file(url, f)


online_retail_obj = fetch_ucirepo(id=352)
online_retail_df = online_retail_obj.data.original
online_retail_dict = online_retail_df.to_dict(orient="records")

# Destination file path to save the downloaded file
absolute_dir_path = os.path.dirname(os.path.realpath(__file__))

# Connect to MongoDB
client = MongoClient(mongodb_uri)
db = client[database_name]


# Create or replace collection with new data
try:
    if collection_name in db.list_collection_names():
        # Delete collection if exists
        db.drop_collection(collection_name)
        print(f"Collection '{collection_name}' replaced with new data.")

    # Create new collection
    collection = db[collection_name]
    collection.insert_many(online_retail_dict)

except Exception as e:
    print(f"An error occurred: {e}")


# Upload the file to Flask server
for f in doc_filenames:
    try:
        upload_url = "http://localhost:8085/upload"

        with open(f, "rb") as file:
            files = {"file": (f, file)}
            response = requests.post(upload_url, files=files)

        print(response)
    except Exception as e:
        print(f"An error occurred: {e}")
# Clean up by deleting the downloaded file
for f in doc_filenames:
    os.remove(f)
print("Downloaded file deleted.")
