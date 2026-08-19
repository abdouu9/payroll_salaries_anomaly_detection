import json

notebook_path = "anomaly_detection.ipynb"
with open(notebook_path, "r", encoding="utf-8") as f:
    nb = json.load(f)

# Search cells for "surrogate" or "surrogate_model"
for idx, cell in enumerate(nb.get("cells", [])):
    cell_type = cell.get("cell_type")
    source = "".join(cell.get("source", []))
    if "surrogate" in source.lower():
        print(f"--- Cell {idx} ({cell_type}) ---")
        # Print up to 1000 characters of the source
        print(source[:1500])
        print("\n" + "="*50 + "\n")
