import json
import sys
import argparse

def load_json(path):
    try:
        with open(path, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Error: {path} not found.")
        return None
    except json.JSONDecodeError:
        print(f"Error: {path} is not valid JSON.")
        return None

def main():
    parser = argparse.ArgumentParser(description="Compare two evaluation JSON files.")
    parser.add_argument("base", help="Base evaluation JSON file")
    parser.add_argument("pr", help="PR evaluation JSON file")
    parser.add_argument("--threshold", type=float, default=0.05, help="Regression threshold (e.g., 0.05 for 5%)")
    parser.add_argument("--output", default="eval_comparison.md", help="Output Markdown file")
    args = parser.parse_args()

    base_data = load_json(args.base) or {"ragas": {}, "simple": {}}
    pr_data = load_json(args.pr) or {"ragas": {}, "simple": {}}

    base_ragas = base_data.get("ragas", {})
    pr_ragas = pr_data.get("ragas", {})
    base_simple = base_data.get("simple", {})
    pr_simple = pr_data.get("simple", {})

    all_metrics = set(base_ragas.keys()).union(pr_ragas.keys()).union(base_simple.keys()).union(pr_simple.keys())

    md_lines = [
        "## Evaluation Score Comparison",
        "",
        "| Metric | Base | PR | Diff | Status |",
        "|---|---|---|---|---|"
    ]

    has_regression = False

    for metric in sorted(all_metrics):
        base_val = base_ragas.get(metric, base_simple.get(metric))
        pr_val = pr_ragas.get(metric, pr_simple.get(metric))

        if isinstance(base_val, (int, float)) and isinstance(pr_val, (int, float)):
            diff = pr_val - base_val
            
            if diff < -args.threshold:
                status = "❌ Regression"
                has_regression = True
            elif diff > 0.01:
                status = "✅ Improved"
            elif diff < 0:
                status = "⚠️ Slight dip"
            else:
                status = "➖ No change"

            if isinstance(base_val, float):
                b_str = f"{base_val:.4f}"
                p_str = f"{pr_val:.4f}"
                d_str = f"{diff:+.4f}"
            else:
                b_str = str(base_val)
                p_str = str(pr_val)
                d_str = f"{diff:+d}"

            md_lines.append(f"| {metric} | {b_str} | {p_str} | {d_str} | {status} |")

    md_content = "\n".join(md_lines)
    
    with open(args.output, "w") as f:
        f.write(md_content)
    
    print(f"Comparison saved to {args.output}")
    print(md_content)

    if has_regression:
        print(f"\\nError: Detected a regression exceeding the threshold of {args.threshold}.")
        sys.exit(1)
    else:
        print("\\nAll metrics passed regression check.")
        sys.exit(0)

if __name__ == "__main__":
    main()
