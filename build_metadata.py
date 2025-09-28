
with open(DATA_PATH, "r") as f:
    data_json = json.load(f)

metadata_json = {}
for state, records in data_json.items():
    for r in records:
        r['state'] = state.strip().upper()
        r['location'] = r['location'].strip().upper()
        # Compute SGWD and category
        r['sgwd'] = (r['ground_water_extraction_ham'] / r['annual_extractable_ground_water_resources_ham']) * 100
        r['approx_category'] = 'Safe' if r['sgwd'] <= 70 else 'Semi-Critical' if r['sgwd'] <= 90 else 'Critical' if r['sgwd'] < 100 else 'Over-Exploited'
        key = f"{r['location']}_{r['state']}_{r['year'].strip()}"
        metadata_json[key] = r