import os
import json
import time
from pathlib import Path

from google import genai
from google.genai import types


MODEL = os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite")
API_KEY = os.environ["GEMINI_API_KEY"]

client = genai.Client(api_key=API_KEY)


FACT_SCHEMA = {
    "type": "object",
    "properties": {
        "facts": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "fact": {
                        "type": "string"
                    },
                    "category": {
                        "type": "string"
                    },
                    "page": {
                        "type": "integer"
                    },
                    "evidence": {
                        "type": "string"
                    }
                },
                "required": [
                    "fact",
                    "category",
                    "page",
                    "evidence"
                ]
            }
        }
    },
    "required": ["facts"]
}


def extract_page_facts(page_number, ocr_text):
    prompt = f"""
तुम Rajasthan competitive exams के लिए FACT EXTRACTOR हो।

यह Sujas PDF का Page {page_number} है।

तुम्हारा काम MCQ बनाना नहीं है।
केवल इसी page पर वास्तव में मौजूद महत्वपूर्ण facts निकालने हैं।

STRICT RULES:

1. केवल दिए गए OCR text से facts निकालो।
2. अपनी तरफ से कोई information मत जोड़ो।
3. अगर OCR अस्पष्ट है तो अनुमान मत लगाओ।
4. तारीख, नाम, पद, स्थान, संख्या, योजना, संस्था आदि बिल्कुल source के अनुसार रखो।
5. हर fact के साथ page number दो।
6. हर fact के साथ छोटा exact evidence दो।
7. एक ही बात को बार-बार fact मत बनाओ।
8. सामान्य/महत्वहीन वाक्य छोड़ दो।
9. ऐसा fact मत बनाओ जिसका आधार इस page के text में नहीं है।
10. MCQ, options या answer मत बनाओ।

Category इनमें से उपयुक्त चुनो:
- नियुक्ति
- योजना
- नीति
- कार्यक्रम
- प्रशासन
- शिक्षा
- स्वास्थ्य
- कृषि
- खेल
- पुरस्कार
- अर्थव्यवस्था
- बजट
- पर्यावरण
- विज्ञान एवं तकनीक
- आधारभूत संरचना
- चुनाव
- व्यक्ति
- स्थान
- अन्य

Page:
{page_number}

OCR TEXT:
{ocr_text}
"""

    for attempt in range(4):
        try:
            response = client.models.generate_content(
                model=MODEL,
                contents=prompt,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema=FACT_SCHEMA,
                    temperature=0.1,
                ),
            )

            data = json.loads(response.text)

            if not isinstance(data, dict):
                return []

            facts = data.get("facts", [])

            valid = []

            for item in facts:
                if not isinstance(item, dict):
                    continue

                fact = str(item.get("fact", "")).strip()
                category = str(item.get("category", "")).strip()
                evidence = str(item.get("evidence", "")).strip()

                try:
                    page = int(item.get("page", page_number))
                except Exception:
                    page = page_number

                if not fact or not evidence:
                    continue

                if page != page_number:
                    continue

                valid.append({
                    "fact": fact,
                    "category": category,
                    "page": page,
                    "evidence": evidence
                })

            return valid

        except Exception as e:
            print(
                f"Page {page_number}: Gemini attempt "
                f"{attempt + 1} failed: {e}"
            )

            if attempt < 3:
                time.sleep(2 ** attempt)

    return []


def deduplicate_facts(facts):
    seen = set()
    result = []

    for fact in facts:
        key = (
            fact["fact"]
            .strip()
            .lower()
            .replace(" ", "")
        )

        if key in seen:
            continue

        seen.add(key)
        result.append(fact)

    return result


def main():
    input_file = Path("ocr_pages.json")
    output_file = Path("verified_facts.json")

    if not input_file.exists():
        raise FileNotFoundError(
            "ocr_pages.json नहीं मिला"
        )

    with open(input_file, "r", encoding="utf-8") as f:
        pages = json.load(f)

    all_facts = []

    print(f"Total pages: {len(pages)}")

    for page in pages:
        page_number = int(page["page"])
        ocr_text = page.get("text", "").strip()

        if not ocr_text:
            print(f"Page {page_number}: empty OCR")
            continue

        print(f"Extracting facts from page {page_number}...")

        facts = extract_page_facts(
            page_number,
            ocr_text
        )

        print(
            f"Page {page_number}: "
            f"{len(facts)} facts found"
        )

        all_facts.extend(facts)

    all_facts = deduplicate_facts(all_facts)

    output = {
        "source": "Rajasthan Sujas",
        "facts_count": len(all_facts),
        "facts": all_facts
    }

    with open(
        output_file,
        "w",
        encoding="utf-8"
    ) as f:
        json.dump(
            output,
            f,
            ensure_ascii=False,
            indent=2
        )

    print()
    print("=" * 50)
    print(f"TOTAL VERIFIED FACTS: {len(all_facts)}")
    print("=" * 50)
    print(f"Saved: {output_file}")


if __name__ == "__main__":
    main()
