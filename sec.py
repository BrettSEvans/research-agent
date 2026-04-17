"""SEC EDGAR client: company lookup, filings discovery, document retrieval."""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable

import httpx
from bs4 import BeautifulSoup

SEC_UA = "Compliance Agent research-agent@saasless.example"
HEADERS = {"User-Agent": SEC_UA, "Accept-Encoding": "gzip, deflate"}

# Explicit connect/read/write/pool timeouts — a single hung socket should not
# hang the whole compliance run.
SEC_TIMEOUT = httpx.Timeout(connect=10.0, read=45.0, write=10.0, pool=10.0)
FETCH_TIMEOUT = httpx.Timeout(connect=10.0, read=90.0, write=10.0, pool=10.0)


@dataclass
class Filing:
    cik: str
    accession: str  # e.g. "0001193125-23-001234"
    form: str  # "10-K", "10-Q", "S-1", "8-K", "D", ...
    filing_date: str
    primary_doc: str  # filename of the primary document
    url: str  # fully-qualified URL to primary_doc


def _cik10(cik: str | int) -> str:
    return str(int(cik)).zfill(10)


def lookup_cik(query: str) -> str | None:
    """Resolve a ticker or company name to a 10-digit CIK."""
    q = query.strip().lower()
    r = httpx.get(
        "https://www.sec.gov/files/company_tickers.json",
        headers=HEADERS,
        timeout=SEC_TIMEOUT,
    )
    r.raise_for_status()
    for row in r.json().values():
        if row["ticker"].lower() == q or row["title"].lower() == q:
            return _cik10(row["cik_str"])
    # fuzzy fallback: substring match on title
    for row in r.json().values():
        if q in row["title"].lower():
            return _cik10(row["cik_str"])
    return None


def list_filings(
    cik: str, forms: Iterable[str] = ("10-K", "10-Q", "S-1", "8-K"), limit: int = 5
) -> list[Filing]:
    """Return the most recent filings for a CIK, filtered by form type."""
    cik10 = _cik10(cik)
    r = httpx.get(
        f"https://data.sec.gov/submissions/CIK{cik10}.json",
        headers=HEADERS,
        timeout=SEC_TIMEOUT,
    )
    r.raise_for_status()
    recent = r.json()["filings"]["recent"]
    accessions = recent["accessionNumber"]
    form_types = recent["form"]
    dates = recent["filingDate"]
    primary_docs = recent["primaryDocument"]

    forms_set = {f.upper() for f in forms}
    out: list[Filing] = []
    for acc, form, date, doc in zip(accessions, form_types, dates, primary_docs):
        if form.upper() not in forms_set:
            continue
        acc_nodash = acc.replace("-", "")
        url = f"https://www.sec.gov/Archives/edgar/data/{int(cik10)}/{acc_nodash}/{doc}"
        out.append(
            Filing(
                cik=cik10,
                accession=acc,
                form=form,
                filing_date=date,
                primary_doc=doc,
                url=url,
            )
        )
        if len(out) >= limit:
            break
    return out


def fetch_text(filing: Filing) -> str:
    """Fetch a filing's primary document and return cleaned plain text."""
    r = httpx.get(
        filing.url, headers=HEADERS, timeout=FETCH_TIMEOUT, follow_redirects=True
    )
    r.raise_for_status()
    ctype = r.headers.get("content-type", "").lower()
    if "html" in ctype or filing.primary_doc.lower().endswith((".htm", ".html")):
        soup = BeautifulSoup(r.text, "html.parser")
        for tag in soup(["script", "style"]):
            tag.decompose()
        text = soup.get_text(separator="\n")
    else:
        text = r.text
    # collapse whitespace
    text = re.sub(r"\n\s*\n+", "\n\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()


def chunk_text(text: str, size: int = 1200, overlap: int = 200) -> list[str]:
    """Character-based chunking with overlap. Crude but reliable."""
    chunks: list[str] = []
    i = 0
    n = len(text)
    while i < n:
        chunks.append(text[i : i + size])
        i += size - overlap
    return [c.strip() for c in chunks if c.strip()]
