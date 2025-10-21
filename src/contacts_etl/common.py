
import re
import uuid
from difflib import SequenceMatcher

# Country/state mappings (subset, extend as needed)
ISO2 = {
    "us":"US","usa":"US","united states":"US","united states of america":"US","u.s.":"US","u.s.a.":"US","america":"US",
    "canada":"CA","ca":"CA","mexico":"MX","mx":"MX","united kingdom":"GB","uk":"GB","u.k.":"GB","great britain":"GB",
    "england":"GB","scotland":"GB","wales":"GB","northern ireland":"GB","ireland":"IE","republic of ireland":"IE",
    "germany":"DE","deutschland":"DE","de":"DE","france":"FR","fr":"FR","italy":"IT","it":"IT","spain":"ES","es":"ES",
    "portugal":"PT","pt":"PT","netherlands":"NL","holland":"NL","nl":"NL","belgium":"BE","be":"BE","switzerland":"CH","ch":"CH",
    "austria":"AT","at":"AT","australia":"AU","au":"AU","new zealand":"NZ","nz":"NZ","india":"IN","in":"IN",
    "china":"CN","cn":"CN","people's republic of china":"CN","prc":"CN","japan":"JP","jp":"JP","south korea":"KR","republic of korea":"KR","kr":"KR",
    "brazil":"BR","br":"BR","argentina":"AR","ar":"AR","south africa":"ZA","za":"ZA","sweden":"SE","se":"SE","norway":"NO","no":"NO",
    "denmark":"DK","dk":"DK","finland":"FI","fi":"FI","czech republic":"CZ","czechia":"CZ","cz":"CZ","poland":"PL","pl":"PL",
    "singapore":"SG","sg":"SG","hong kong":"HK","hk":"HK","israel":"IL","il":"IL","united arab emirates":"AE","uae":"AE","ae":"AE"
}
STATE_ABBR = {
    "alabama":"AL","alaska":"AK","arizona":"AZ","arkansas":"AR","california":"CA","colorado":"CO",
    "connecticut":"CT","delaware":"DE","florida":"FL","georgia":"GA","hawaii":"HI","idaho":"ID",
    "illinois":"IL","indiana":"IN","iowa":"IA","kansas":"KS","kentucky":"KY","louisiana":"LA",
    "maine":"ME","maryland":"MD","massachusetts":"MA","michigan":"MI","minnesota":"MN","mississippi":"MS",
    "missouri":"MO","montana":"MT","nebraska":"NE","nevada":"NV","new hampshire":"NH","new jersey":"NJ",
    "new mexico":"NM","new york":"NY","north carolina":"NC","north dakota":"ND","ohio":"OH","oklahoma":"OK",
    "oregon":"OR","pennsylvania":"PA","rhode island":"RI","south carolina":"SC","south dakota":"SD",
    "tennessee":"TN","texas":"TX","utah":"UT","vermont":"VT","virginia":"VA","washington":"WA",
    "west virginia":"WV","wisconsin":"WI","wyoming":"WY","district of columbia":"DC","dc":"DC"
}
PARTICLES = {"da","de","del","della","der","di","la","le","van","von","den","ten","ter","du","st","st.","san","mac","mc","o","d","l"}

def normalize_state(s:str)->str:
    s = (s or "").strip()
    if not s: return ""
    if len(s)==2 and s.isalpha(): return s.upper()
    return STATE_ABBR.get(s.lower(), s.upper())

def normalize_country_iso2(c:str)->str:
    c = (c or "").strip()
    if not c: return ""
    return ISO2.get(c.lower(), c.upper() if len(c)==2 else c)

def normalize_prof_token(tok:str)->str:
    return re.sub(r"[^a-z0-9]", "", (tok or "").lower())

def parse_name_multi_last(name_str:str):
    """Parse first/middle/last considering multi-element surnames and apostrophes."""
    if not name_str: return "","",""
    tokens = name_str.split()
    if len(tokens) == 1: return tokens[0], "", ""
    last_parts = [tokens[-1]]
    i = len(tokens) - 2
    while i >= 1:
        t = tokens[i]
        t_clean = (t or "").lower().strip(".")
        if t_clean in PARTICLES or (t_clean in {"o","d","l"} and i+1 < len(tokens) and "'" in tokens[i+1]):
            last_parts.insert(0, t); i -= 1; continue
        if t and t[0].islower() and tokens[i+1][0].isupper():
            last_parts.insert(0, t); i -= 1; continue
        break
    first = tokens[0]
    middle = " ".join(tokens[1: i+1]) if i >= 1 else ""
    last = " ".join(last_parts)
    return first, middle, last

def strip_suffixes_and_parse_name(full_name:str, gen_suffixes:set, prof_suffixes:set):
    """Return (first, middle, last, gen_suffix, prof_suffixes_list, maiden, full_name_clean)."""
    if not full_name or str(full_name).strip() == "":
        return "", "", "", "", [], "", ""
    name = str(full_name).strip()
    maiden = ""
    paren = re.search(r"\(([^)]+)\)", name)
    if paren:
        maiden = paren.group(1).strip()
        name = (name[:paren.start()] + name[paren.end():]).strip()
    parts = [p.strip() for p in re.split(r"[,\u2013\u2014-]+", name) if p.strip()]
    kept_parts = []
    gen_suffix = ""
    prof_out = []
    def is_prof_token(token:str)->bool:
        t = normalize_prof_token(token)
        return (t in prof_suffixes) or (t.endswith("spc6"))
    for p in parts:
        tokens = p.split()
        if len(tokens) == 1:
            tok = tokens[0]
            if normalize_prof_token(tok) in gen_suffixes:
                gen_suffix = p; continue
            if is_prof_token(tok):
                prof_out.append(p); continue
            kept_parts.append(p); continue
        trailing = []
        while tokens and is_prof_token(tokens[-1]):
            trailing.append(tokens.pop())
        if trailing: prof_out.extend(trailing[::-1])
        if len(tokens) == 1 and normalize_prof_token(tokens[0]) in gen_suffixes:
            gen_suffix = tokens[0]; tokens = []
        if tokens: kept_parts.append(" ".join(tokens))
    base = " ".join(kept_parts).strip()
    first, middle, last = parse_name_multi_last(base)
    full_name_clean = " ".join([t for t in [first, middle, last, gen_suffix] if t]).strip()
    return first, middle, last, gen_suffix, prof_out, maiden, full_name_clean

def deterministic_uuid(namespace_str:str)->str:
    ns = uuid.UUID("12345678-1234-5678-1234-567812345678")
    return str(uuid.uuid5(ns, namespace_str))

def seq_ratio(a:str,b:str)->float:
    return SequenceMatcher(None, (a or "").lower(), (b or "").lower()).ratio()
