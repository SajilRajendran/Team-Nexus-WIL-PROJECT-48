# Data sources and collection

**Owner:** Rohit Devaraj (Data Preprocessing & Retrieval Engineer)
**Full list:** `data/sources.csv`

## Why a curated set
We used a small set of **official government sources** instead of crawling the web, so every answer can be traced to a trusted page and the knowledge base is easy to justify.

## What is in the set (30 planned, 29 indexed)
| Group | Sources |
|---|---|
| Program-wide pages (all centres) | 14 |
| Pilot program pages | 2 |
| Individual pilot centres (Broadmeadows, Camberwell Junction, Chadstone, Epping, Frankston, Moorabbin, Niddrie / North Essendon, Preston, Ringwood) | 9 |
| Council and other pages (Box Hill, Glen Waverley, Glen Eira) | 5 |

- **Publishers:** Planning Victoria (23), councils (Monash, Whitehorse, Glen Eira), Victorian Planning Authority, Victorian Government.
- **Formats:** 26 HTML pages, 4 PDFs (including the VicSmart Guide).

## What we record for each source
`doc_id`, title, URL, activity centre, publisher, document type, local file name, **date retrieved** and **SHA-256 hash**.
The date and hash are filled in by the download script, so each answer can be traced to a specific version of a page.

## Processing
- HTML cleaned with BeautifulSoup; PDFs extracted with PyMuPDF / pdfplumber.
- Text split into ~300-word chunks with 60-word overlap → **522 chunks**.
- Each chunk keeps its title, URL, centre and page number, which makes citations possible.

## Known gaps
- **Whitehorse Amendment C175** failed to download, so Box Hill relies on the remaining council page.
- Coverage is uneven: only some of the 60 centres have their own page.
- No public-transport timetable data yet (needed for the student user story in Milestone 1).
