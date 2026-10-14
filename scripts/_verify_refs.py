import json, re, sys, time, urllib.request, xml.etree.ElementTree as ET
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
DOIS = {"cook2013casas": "10.1109/MC.2012.328", "abadi2016dpsgd": "10.1145/2976749.2978318", "dwork2014algorithmic": "10.1561/0400000042", "mcsherry2007exponential": "10.1109/FOCS.2007.66",
        "mironov2017rdp": "10.1109/CSF.2017.11", "ashraedb2018": "10.1016/j.buildenv.2018.06.022", "miller2020bdg2": "10.1038/s41597-020-00712-x"}
ARX = {"mcmahan2017fedavg": "1602.05629", "yousefpour2021opacus": "2109.12298", "carlini2022lira": "2112.03570", "sinergym2024": "2412.08293", "jia2016privacy": "1607.03140",
       "fedhvac2024": "2405.00389", "encryptedhvac2023": "2312.05183", "dppfeddsu2025": "2503.09192"}
def get(u):
    for i in range(4):
        try: return urllib.request.urlopen(urllib.request.Request(u, headers={"User-Agent": "palace-ref-check (mailto:deerajsk2006@gmail.com)"}), timeout=30).read()
        except Exception as e: err = e; time.sleep(2 ** i)
    raise err
bib, bad = [], []
for k, d in DOIS.items():
    try:
        m = json.loads(get(f"https://api.crossref.org/works/{d}"))["message"]; t = m["title"][0]; a = " and ".join(f"{x.get('family','')}, {x.get('given','')}" for x in m.get("author", [])[:12]); y = (m.get("issued", {}).get("date-parts") or [[None]])[0][0]
        bib.append(f"@misc{{{k},\n  title = {{{t}}},\n  author = {{{a}}},\n  year = {{{y}}},\n  doi = {{{d}}},\n  note = {{verified via Crossref}}\n}}\n"); print("OK ", k, y, t[:70])
    except Exception as e: bad.append((k, d, repr(e))); print("FAIL", k, e)
for k, i in ARX.items():
    try:
        r = ET.fromstring(get(f"http://export.arxiv.org/api/query?id_list={i}")); ns = {"a": "http://www.w3.org/2005/Atom"}; e = r.find("a:entry", ns); t = re.sub(r"\s+", " ", e.find("a:title", ns).text.strip())
        if t.lower().startswith("error"): raise ValueError("arXiv error entry")
        a = " and ".join(x.find("a:name", ns).text for x in e.findall("a:author", ns)[:12]); y = e.find("a:published", ns).text[:4]
        bib.append(f"@misc{{{k},\n  title = {{{t}}},\n  author = {{{a}}},\n  year = {{{y}}},\n  eprint = {{{i}}},\n  archivePrefix = {{arXiv}},\n  note = {{verified via arXiv API}}\n}}\n"); print("OK ", k, y, t[:70])
    except Exception as e_: bad.append((k, i, repr(e_))); print("FAIL", k, e_)
    time.sleep(3)
(ROOT / "results").mkdir(exist_ok=True); (ROOT / "results/references_verified.bib").write_text("\n".join(bib))
if bad: open(ROOT / "logs/ISSUES.md", "a").write("".join(f"- reference NOT verified (left out): {k} {i} {err}\n" for k, i, err in bad))
