import pandas as pd
import requests
import time
import json
import os
import xml.etree.ElementTree as ET

BASE_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/" # It is the basic address that is common to all endpoints of a API.
RATE_LIMIT_DELAY = 0.34  # 3 request/sec

# API => BASE URL + ENDPOINT(esearch + efetch)

# Stats
errors = 0
duplicates_removed = 0


def safe_request(url, retries=3):    # Secure API call
    global errors
    for _ in range(retries):
        try:
            response = requests.get(url, timeout=10) # HTTP Get request, send the data from the URL.
            response.raise_for_status()
            return response
        except requests.exceptions.RequestException:
            errors += 1
            time.sleep(1)
    return None

def get_paper_ids(term):  # Find article IDs (PMID)
    url = f"{BASE_URL}esearch.fcgi?db=pubmed&term={term}&retmax=5&retmode=json&sort=pub+date"   # esearch, bring up to 5 results (retmax=5)
    response = safe_request(url)
    if response:
        return response.json().get("esearchresult", {}).get("idlist", []) # search idlist key
    return []

def extract_text(element, path): # for error protection
    found = element.find(path)
    return found.text.strip() if found is not None and found.text else None

def fetch_article_details(pmid): # extract article details
    global errors

    url = f"{BASE_URL}efetch.fcgi?db=pubmed&id={pmid}&retmode=xml"   # efetch, provide all the details of the article with this ID
    response = safe_request(url)
    if not response:
        return None

    try:
        root = ET.fromstring(response.content)
        article = root.find(".//PubmedArticle")

        if article is None:
            return None

        title = extract_text(article, ".//ArticleTitle")

        abstract_elems = article.findall(".//AbstractText")
        abstract = " ".join([elem.text for elem in abstract_elems if elem.text]) if abstract_elems else None

        author_elem = article.find(".//Author")
        if author_elem is not None:
            last = extract_text(author_elem, "LastName")
            fore = extract_text(author_elem, "ForeName")
            first_author = f"{fore} {last}" if fore and last else last or fore
        else:
            first_author = None

        journal = extract_text(article, ".//Journal/Title")
        year = extract_text(article, ".//PubDate/Year")

        doi = None
        for id_elem in article.findall(".//ArticleId"):
            if id_elem.attrib.get("IdType") == "doi":
                doi = id_elem.text
                break

        return {
            "pmid": pmid,
            "title": title,
            "abstract": abstract,
            "authors": first_author,
            "journal": journal,
            "year": year,
            "doi": doi,
            "matched_terms": []
        }

    except Exception:
        errors += 1
        return None


def main():
    global duplicates_removed

    # ✅ Path fix
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    csv_path = os.path.join(BASE_DIR, "data", "medical_terms.csv")
    output_path = os.path.join(BASE_DIR, "data", "output.json")

    # ✅ Read CSV
    terms = pd.read_csv(csv_path)["term"].tolist()

    unique_articles = {}

    for term in terms:
        ids = get_paper_ids(term)    # Fetch the ID that related with term
        time.sleep(RATE_LIMIT_DELAY) # Pauses the program briefly

        for pmid in ids:
            if pmid not in unique_articles:
                article_data = fetch_article_details(pmid) # fetch pmid, title,abstract,author,journal,year and DOI.
                time.sleep(RATE_LIMIT_DELAY)

                if article_data:
                    article_data["matched_terms"].append(term)
                    unique_articles[pmid] = article_data
            else:
                duplicates_removed += 1  # If the article is alrady exist add 1
                if term not in unique_articles[pmid]["matched_terms"]:
                    unique_articles[pmid]["matched_terms"].append(term)

    # ✅ JSON save (path fix)
    output = list(unique_articles.values())
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    # ✅ Summary
    print("----- SUMMARY -----")
    print(f"Terms processed: {len(terms)}")
    print(f"Unique articles: {len(unique_articles)}")
    print(f"Duplicates removed: {duplicates_removed}")
    print(f"Errors: {errors}")


if __name__ == "__main__":
    main()