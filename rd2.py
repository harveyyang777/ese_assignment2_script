import os
import requests
import pandas as pd
import numpy as np
from datetime import datetime
from scipy.stats import spearmanr
import matplotlib.pyplot as plt

# -------------------------
# CONFIGURATION
# -------------------------
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")
HEADERS = {"Authorization": f"token {GITHUB_TOKEN}"}

PROJECTS = [
    "thingsboard/thingsboard",
    "apache/dolphinscheduler",
    "wireapp/wire-server",
    "keptn/keptn",
    "open-metadata/OpenMetadata",
    "dapr/dapr",
    "jaegertracing/jaeger",
    "appwrite/appwrite",
    "camunda/zeebe",
    "gravitee-io/gravitee-api-management"
]

# -------------------------
# GITHUB HELPERS
# -------------------------
def gh_get(url, params=None):
    r = requests.get(url, headers=HEADERS, params=params)
    r.raise_for_status()
    return r.json()

def list_files(repo):
    owner, name = repo.split("/")
    tree = gh_get(
        f"https://api.github.com/repos/{owner}/{name}/git/trees/HEAD",
        params={"recursive": "1"}
    )
    return [f["path"].lower() for f in tree["tree"] if f["type"] == "blob"]

def classify_tests(files):
    test_files = [f for f in files if "test" in f]
    unit_files = [
        f for f in test_files
        if "unit" in f or f.endswith("_test.go") or "src/test" in f
    ]
    return len(unit_files), len(test_files)

def collect_bug_resolution_times(repo):
    owner, name = repo.split("/")
    issues = gh_get(
        f"https://api.github.com/repos/{owner}/{name}/issues",
        params={"state": "closed", "labels": "bug", "per_page": 100}
    )

    times = []
    for issue in issues:
        if "pull_request" in issue:
            continue
        created = datetime.fromisoformat(issue["created_at"].replace("Z",""))
        closed = datetime.fromisoformat(issue["closed_at"].replace("Z",""))
        times.append((closed - created).days)
    return times

# -------------------------
# DATA COLLECTION
# -------------------------
rows = []

for repo in PROJECTS:
    files = list_files(repo)
    unit, total = classify_tests(files)
    bug_times = collect_bug_resolution_times(repo)

    rows.append({
        "project": repo,
        "unit_presence": int(unit > 0),
        "unit_ratio": unit / total if total > 0 else np.nan,
        "median_bug_resolution_days": np.median(bug_times) if bug_times else np.nan,
        "bug_count": len(bug_times)
    })

df = pd.DataFrame(rows)
df.to_csv("rq2_dataset.csv", index=False)

# -------------------------
# SCATTER PLOT
# -------------------------
clean = df.dropna(subset=["unit_ratio", "median_bug_resolution_days"])

plt.figure()
plt.scatter(
    clean["unit_ratio"],
    clean["median_bug_resolution_days"]
)
plt.xlabel("Unit Test Ratio")
plt.ylabel("Median Bug Resolution Time (days)")
plt.title("Unit Test Emphasis vs Defect Resolution Time")

plt.savefig("unit_ratio_vs_bug_resolution.png", dpi=300)
plt.show()

# -------------------------
# CORRELATION
# -------------------------
rho, p = spearmanr(
    clean["unit_ratio"],
    clean["median_bug_resolution_days"]
)

print("Spearman rho:", rho)
print("p-value:", p)
