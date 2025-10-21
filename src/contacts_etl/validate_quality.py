
import os, re, csv, json, argparse
import pandas as pd

# Optional libs
try:
    from email_validator import validate_email, EmailNotValidError
    HAS_EMAIL_VALIDATOR=True
except Exception:
    HAS_EMAIL_VALIDATOR=False

try:
    import phonenumbers
    HAS_PHONENUMBERS=True
except Exception:
    HAS_PHONENUMBERS=False

try:
    import usaddress
    HAS_USADDRESS=True
except Exception:
    HAS_USADDRESS=False

def pct(n, d): 
    return round((n / d * 100.0), 2) if d else 0.0

def validate_email_list(emails_field, syntax_only=True, dns_mx=False):
    details = []
    if not isinstance(emails_field, str) or not emails_field.strip(): return 0, 0, details
    parts = [p for p in emails_field.split("|") if p.strip()]
    valid_count = 0
    for p in parts:
        email = p.split("::")[0].strip()
        label = p.split("::")[1].strip() if "::" in p else ""
        is_valid = False
        if HAS_EMAIL_VALIDATOR:
            try:
                normalized = validate_email(
                    email,
                    check_deliverability = True if dns_mx else False
                ).normalized if not syntax_only or dns_mx else validate_email(email, check_deliverability=False).normalized
                email = normalized
                is_valid = True
            except EmailNotValidError:
                is_valid = False
        else:
            is_valid = bool(re.match(r"^[A-Za-z0-9._%+\-']+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}$", email))
        valid_count += 1 if is_valid else 0
        details.append({"email": email, "label": label, "valid": is_valid})
    return valid_count, len(parts), details

def validate_phone_list(phones_field):
    details = []
    if not isinstance(phones_field, str) or not phones_field.strip(): return 0, 0, details
    parts = [p for p in phones_field.split("|") if p.strip()]
    valid_count = 0
    for p in parts:
        phone = p.split("::")[0].strip()
        label = p.split("::")[1].strip() if "::" in p else ""
        is_valid = False
        if HAS_PHONENUMBERS:
            try:
                pn = phonenumbers.parse(phone, None if phone.startswith("+") else "US")
                is_valid = phonenumbers.is_possible_number(pn) and phonenumbers.is_valid_number(pn)
            except Exception:
                is_valid = False
        else:
            is_valid = phone.startswith("+") and len(re.sub(r"\D", "", phone)) >= 11
        valid_count += 1 if is_valid else 0
        details.append({"phone": phone, "label": label, "valid": is_valid})
    return valid_count, len(parts), details

def parse_addresses_json(addresses_json):
    details = []
    if not isinstance(addresses_json, str) or not addresses_json.strip(): return 0, 0, details
    try:
        addrs = json.loads(addresses_json)
    except Exception:
        return 0, 0, details
    valid_count = 0
    for a in addrs:
        street = (a.get("street","") or "").strip()
        city = (a.get("city","") or "").strip()
        state = (a.get("state","") or "").strip()
        postal = (a.get("postal_code","") or "").strip()
        country = (a.get("country","") or "").strip()
        is_valid = bool(street) and bool(city or postal)
        details.append({
            "street": street, "city": city, "state": state, "postal_code": postal,
            "country": country, "valid": is_valid
        })
        valid_count += 1 if is_valid else 0
    return valid_count, len(addrs), details

def main():
    import yaml
    parser = argparse.ArgumentParser(description="Validate & score consolidated contacts.")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--contacts-csv", type=str, default=None)
    parser.add_argument("--out-dir", type=str, default=None)
    parser.add_argument("--email-dns-mx", action="store_true", help="Enable DNS/MX deliverability check in email-validator (requires internet).")

    args = parser.parse_args()
    cfg = {}
    if args.config:
        with open(args.config, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
    validation_cfg = cfg.get("validation", {})
    outputs = cfg.get("outputs", {})

    contacts_csv = args.contacts_csv or os.path.join(outputs.get("dir", os.getcwd()), "consolidated_contacts.csv")
    out_dir = args.out_dir or outputs.get("dir", os.getcwd())
    dns_mx = args.email_dns_mx or validation_cfg.get("email_dns_mx_check", False)

    df = pd.read_csv(contacts_csv, dtype=str, keep_default_na=False, quoting=csv.QUOTE_ALL)

    records = []
    for _, row in df.iterrows():
        e_valid, e_total, e_details = validate_email_list(row.get("emails",""), syntax_only=True, dns_mx=dns_mx)
        p_valid, p_total, p_details = validate_phone_list(row.get("phones",""))
        a_valid, a_total, a_details = parse_addresses_json(row.get("addresses_json",""))
        rec = {
            "contact_id": row.get("contact_id",""),
            "full_name": row.get("full_name",""),
            "company": row.get("company",""),
            "title": row.get("title",""),
            "linkedin_url": row.get("linkedin_url",""),
            "email_valid_count": e_valid, "email_total": e_total,
            "phone_valid_count": p_valid, "phone_total": p_total,
            "addr_valid_count": a_valid, "addr_total": a_total,
            "emails_detail": json.dumps(e_details, ensure_ascii=False),
            "phones_detail": json.dumps(p_details, ensure_ascii=False),
            "addresses_detail": json.dumps(a_details, ensure_ascii=False)
        }
        # Simple score (can parametrize later)
        email_score = 40 if (0 < e_total == e_valid) else (20 if e_valid > 0 else 0)
        phone_score = 30 if (0 < p_total == p_valid) else (15 if p_valid > 0 else 0)
        addr_score  = 30 if a_valid>0 else 0
        rec["quality_score"] = email_score + phone_score + addr_score
        records.append(rec)

    validation_df = pd.DataFrame(records)
    out_validation = os.path.join(out_dir, "validation_report.csv")
    validation_df.to_csv(out_validation, index=False, encoding="utf-8", quoting=csv.QUOTE_ALL)

    scored_df = df.merge(validation_df[["contact_id","email_valid_count","email_total",
                                        "phone_valid_count","phone_total","addr_valid_count",
                                        "addr_total","quality_score"]],
                         on="contact_id", how="left")
    out_scored = os.path.join(out_dir, "contact_quality_scored.csv")
    scored_df.to_csv(out_scored, index=False, encoding="utf-8", quoting=csv.QUOTE_ALL)

    total = len(scored_df)
    has_any_email = sum(1 for x in scored_df["email_total"] if str(x).strip() and int(x or 0) > 0)
    has_any_phone = sum(1 for x in scored_df["phone_total"] if str(x).strip() and int(x or 0) > 0)
    has_any_addr  = sum(1 for x in scored_df["addr_total"] if str(x).strip() and int(x or 0) > 0)
    print({
        "contacts_total": total,
        "has_any_email_pct": round(has_any_email/total*100,2) if total else 0,
        "has_any_phone_pct": round(has_any_phone/total*100,2) if total else 0,
        "has_any_address_pct": round(has_any_addr/total*100,2) if total else 0
    })
    print(f"Saved: {out_validation}")
    print(f"Saved: {out_scored}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
