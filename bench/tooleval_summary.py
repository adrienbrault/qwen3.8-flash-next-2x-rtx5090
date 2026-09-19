#!/usr/bin/env python3
"""Summarize a tool-eval-bench json. usage: tooleval_summary.py FILE [TAG]"""
import json, sys
d = json.load(open(sys.argv[1])); tag = sys.argv[2] if len(sys.argv) > 2 else ""
t = d["trial_statistics"]; ps = t.get("per_scenario", {}); k = len(next(iter(ps.values()))["points"]) if ps else 0
pts = [sum(v["points"][i] for v in ps.values()) for i in range(k)]
lost = [c["category"] + ":" + str(c["earned"]) + "/" + str(c["max"]) for c in d["scores"]["category_scores"] if c["earned"] < c["max"]]
print(f"{tag} tool-eval 69x{k}: {t['final_score_mean']} +- {t['final_score_stddev']} CI {t.get('final_score_ci95')} points {pts} {lost}")
