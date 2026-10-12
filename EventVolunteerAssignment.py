# Group 17 - Event Volunteer Assignment System
# Group Members/Roles:
#   Rafael Castillo - KNIME Workflow Developer
#   James Diaz - Data Analyst
#   Kelzo Lumabas - Python Developer
#   Benedict Roxas - Streamlit Developer

# Primary Algorithm: Weighted Scoring with Hungarian Assignment Optimization 
# Comparison Algorithm: Rule-Based Matching (Greedy Approach)

# Install dependencies in terminal: pip install pandas numpy scipy

# Run this script with the datasets folder containing:
#    EventVolunteerList - Sheet1.csv
#    TaskList - Sheet1.csv

# Outputs will be saved in the 'outputs' folder:
#    ranked_candidates.csv
#    weighted_assignments.csv
#    rule_based_assignments.csv
#    unfilled_tasks.csv

# IMPORTANT NOTES:
# All data in all datasets are synthetic demonstration data
# The given weights for each criterion are arbitrary and can be adjusted in the WEIGHTS dictionary
# The scoring system is designed to prioritize eligible volunteers based on weighted criteria, but it does not guarantee that all tasks will be filled, especially if there are insufficient eligible volunteers
# The Hungarian algorithm is used to maximize the number of eligible assignments and then the total weighted score, while the greedy rule-based matching serves as a baseline for comparison
# It assumes the Event is held in one place, and that volunteers are close to that area. Distances are in kilometers

from pathlib import Path
from datetime import datetime
import re
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "datasets"
OUTPUT_DIR = BASE_DIR / "outputs"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

VOLUNTEER_FILE = DATA_DIR / "EventVolunteerList - Sheet1.csv"
TASK_FILE = DATA_DIR / "TaskList - Sheet1.csv"

# Weights - These must add up to 1.00 (100%)
WEIGHTS = {
    "category": 0.35,
    "proficiency": 0.25,
    "availability": 0.20,
    "speed": 0.10,
    "distance": 0.10,
}

PROFICIENCY_SCORE = {
    "beginner": 40,
    "intermediate": 60,
    "advanced": 80,
    "expert": 100,
}

SPEED_SCORE = {
    "slow": 40,
    "moderate": 70,
    "fast": 100,
}

MAX_DISTANCE_KM = 5.0

# Convert a time such as '8:00 AM' into minutes after midnight
def parse_time(value):
    value = str(value).strip().upper().replace(".", "")
    for fmt in ("%I:%M %p", "%I %p"):
        try:
            parsed = datetime.strptime(value, fmt)
            return parsed.hour * 60 + parsed.minute
        except ValueError:
            pass
    raise ValueError(f"Could not understand time: {value!r}")

# Parse a range such as '8:00 AM – 12:00 PM' or '8:00 AM-11:00 AM'
def parse_time_range(value):
    value = str(value).strip()
    # Normalize different dash characters to a normal hyphen
    value = re.sub(r"[–—−]", "-", value)
    parts = value.split("-")
    if len(parts) != 2:
        raise ValueError(f"Could not understand time range: {value!r}")
    start = parse_time(parts[0])
    end = parse_time(parts[1])
    if end <= start:
        raise ValueError(f"End time must be after start time: {value!r}")
    return start, end

# Normalize text for case-insensitive comparisons
def clean_text(value):
    if pd.isna(value):
        return ""
    return str(value).strip().casefold()

# Load CSV files
def load_and_validate_data():
    if not VOLUNTEER_FILE.exists():
        raise FileNotFoundError(f"Volunteer CSV Not Found: {VOLUNTEER_FILE}")
    if not TASK_FILE.exists():
        raise FileNotFoundError(f"Task CSV Not Found: {TASK_FILE}")

    volunteers = pd.read_csv(VOLUNTEER_FILE)
    tasks = pd.read_csv(TASK_FILE)

    # Clean accidental spaces in column headers
    volunteers.columns = [str(c).strip() for c in volunteers.columns]
    tasks.columns = [str(c).strip() for c in tasks.columns]

    # Ensure numeric fields are numeric and valid
    volunteers["Volunteer ID"] = pd.to_numeric(
        volunteers["Volunteer ID"], errors="raise"
    ).astype(int)
    tasks["Task ID"] = pd.to_numeric(tasks["Task ID"], errors="raise").astype(int)
    tasks["Required Time (hrs)"] = pd.to_numeric(
        tasks["Required Time (hrs)"], errors="raise"
    )
    tasks["Distance to Task (km)"] = pd.to_numeric(
        tasks["Distance to Task (km)"], errors="raise"
    )

    if volunteers["Volunteer ID"].duplicated().any():
        raise ValueError("Volunteer IDs must be unique.")
    if tasks["Task ID"].duplicated().any():
        raise ValueError("Task IDs must be unique.")
    if (tasks["Required Time (hrs)"] <= 0).any():
        raise ValueError("Required task hours must be greater than zero.")
    if (tasks["Distance to Task (km)"] < 0).any():
        raise ValueError("Task distance cannot be negative.")

    for column in ["Type", "Proficiency", "Speed", "Available Hours"]:
        if volunteers[column].isna().any() or (volunteers[column].astype(str).str.strip() == "").any():
            raise ValueError(f"Volunteer field '{column}' has missing values.")

    for column in ["Task Description", "Task Category", "Schedule"]:
        if tasks[column].isna().any() or (tasks[column].astype(str).str.strip() == "").any():
            raise ValueError(f"Task field '{column}' has missing values.")

    # Parse schedule fields once so later comparisons are consistent
    volunteer_ranges = volunteers["Available Hours"].apply(parse_time_range)
    volunteers["Availability Start"] = volunteer_ranges.apply(lambda x: x[0])
    volunteers["Availability End"] = volunteer_ranges.apply(lambda x: x[1])
    volunteers["Availability Duration (hrs)"] = (
        volunteers["Availability End"] - volunteers["Availability Start"]
    ) / 60

    task_ranges = tasks["Schedule"].apply(parse_time_range)
    tasks["Task Start"] = task_ranges.apply(lambda x: x[0])
    tasks["Task End"] = task_ranges.apply(lambda x: x[1])
    tasks["Schedule Duration (hrs)"] = (
        tasks["Task End"] - tasks["Task Start"]
    ) / 60

    return volunteers, tasks

# Return eligibility, criterion scores, total score, and explanation
def score_pair(volunteer, task):
    reasons = []

    category_match = (
        clean_text(volunteer["Type"]) == clean_text(task["Task Category"])
    )
    category_score = 100.0 if category_match else 0.0

    proficiency_key = clean_text(volunteer["Proficiency"])
    speed_key = clean_text(volunteer["Speed"])
    proficiency_score = float(PROFICIENCY_SCORE.get(proficiency_key, 0))
    speed_score = float(SPEED_SCORE.get(speed_key, 0))

    # A volunteer must be available for the complete task schedule
    covers_schedule = (
        volunteer["Availability Start"] <= task["Task Start"]
        and volunteer["Availability End"] >= task["Task End"]
    )
    enough_hours = (
        volunteer["Availability Duration (hrs)"] >= task["Required Time (hrs)"]
    )
    availability_eligible = covers_schedule and enough_hours

    # Eligible volunteers receive an availability score based on how much availability time they have relative to the required task hours (capped at 100)
    if availability_eligible:
        availability_score = min(
            100.0,
            (volunteer["Availability Duration (hrs)"] /
             float(task["Required Time (hrs)"])) * 100.0
        )
    else:
        availability_score = 0.0
        if not covers_schedule:
            reasons.append("Does not cover the complete task schedule")
        if not enough_hours:
            reasons.append("Insufficient total available hours")

    # Volunteer-specific distance: nearer volunteers receive higher scores
    distance_km = float(volunteer["Distance from Event (km)"])
    distance_score = max(0.0, 100.0 * (1.0 - distance_km / MAX_DISTANCE_KM))

    eligible = availability_eligible
    if not category_match:
        reasons.append("Task category does not match volunteer category")
    if proficiency_score == 0:
        reasons.append("Unrecognized proficiency value")
    if speed_score == 0:
        reasons.append("Unrecognized speed value")

    total_score = (
        category_score * WEIGHTS["category"]
        + proficiency_score * WEIGHTS["proficiency"]
        + availability_score * WEIGHTS["availability"]
        + speed_score * WEIGHTS["speed"]
        + distance_score * WEIGHTS["distance"]
    )

    # Eligibility is separate from score. A high score cannot override schedule conflict
    explanation = "Eligible; ranked by weighted score" if eligible else "; ".join(reasons)

    return {
        "Category Match Score": category_score,
        "Proficiency Score": proficiency_score,
        "Availability Score": availability_score,
        "Speed Score": speed_score,
        "Distance Score": round(distance_score, 2),
        "Weighted Score": round(total_score, 2),
        "Eligible": eligible,
        "Explanation": explanation,
        "Category Match": category_match,
        "Availability Eligible": availability_eligible,
        "Distance from Event (km)": distance_km,
        "Distance to Task (km)": float(task["Distance to Task (km)"]),
    }

# Create one row for every possible volunteer-task pair
def build_candidate_table(volunteers, tasks):
    rows = []
    for _, task in tasks.iterrows():
        for _, volunteer in volunteers.iterrows():
            scores = score_pair(volunteer, task)
            rows.append({
                "Task ID": task["Task ID"],
                "Task Description": task["Task Description"],
                "Task Category": task["Task Category"],
                "Task Schedule": task["Schedule"],
                "Required Time (hrs)": task["Required Time (hrs)"],
                "Volunteer ID": volunteer["Volunteer ID"],
                "Volunteer Name": volunteer["Name"],
                "Volunteer Skill": volunteer["Skills"],
                "Volunteer Category": volunteer["Type"],
                "Proficiency": volunteer["Proficiency"],
                "Speed": volunteer["Speed"],
                "Available Hours": volunteer["Available Hours"],
                **scores,
            })

    candidates = pd.DataFrame(rows)
    candidates = candidates.sort_values(
        by=["Task ID", "Eligible", "Weighted Score", "Proficiency Score"],
        ascending=[True, False, False, False],
        kind="stable",
    )
    return candidates

# Maximize the number of eligible assignments, then total weighted score
# Each task receives at most one volunteer, and each volunteer receives at most one task
def assign_hungarian(candidates):
    if candidates.empty:
        return pd.DataFrame(columns=candidates.columns)

    task_ids = sorted(candidates["Task ID"].unique())
    volunteer_ids = sorted(candidates["Volunteer ID"].unique())
    task_count, volunteer_count = len(task_ids), len(volunteer_ids)

    # One dummy column per task means any task may be left unfilled
    scores = np.zeros((task_count, volunteer_count + task_count), dtype=float)
    scores[:, :volunteer_count] = -1_000_000_000.0
    task_index = {value: i for i, value in enumerate(task_ids)}
    volunteer_index = {value: i for i, value in enumerate(volunteer_ids)}
    candidate_lookup = {}

    # A 1000-point bonus ensures valid assignment count is prioritized over the smaller 0-100 weighted-score differences
    for _, row in candidates.iterrows():
        task_id, volunteer_id = int(row["Task ID"]), int(row["Volunteer ID"])
        candidate_lookup[(task_id, volunteer_id)] = row
        if bool(row["Eligible"]):
            scores[task_index[task_id], volunteer_index[volunteer_id]] = (
                1000.0 + float(row["Weighted Score"])
            )

    task_indices, assigned_indices = linear_sum_assignment(scores, maximize=True)
    assignments = []
    for task_idx, assigned_idx in zip(task_indices, assigned_indices):
        if assigned_idx >= volunteer_count:
            continue  # assigned to a dummy column: task remains unfilled
        task_id = task_ids[task_idx]
        volunteer_id = volunteer_ids[assigned_idx]
        row = candidate_lookup[(task_id, volunteer_id)]
        if bool(row["Eligible"]):
            item = row.to_dict()
            item["Explanation"] = ("Eligible; selected by Hungarian optimization to maximize the number of assignments and then total weighted score")
            assignments.append(item)

    if not assignments:
        return pd.DataFrame(columns=candidates.columns)
    return pd.DataFrame(assignments).sort_values("Task ID").reset_index(drop=True)

# Comparison baseline: greedy rule-based matching, one task per volunteer
def assign_rule_based_greedy(candidates):
    eligible = candidates[candidates["Eligible"]].copy()
    eligible = eligible.sort_values(
        ["Task ID", "Category Match", "Proficiency Score", "Availability Score",
         "Speed Score", "Distance Score", "Volunteer ID"],
        ascending=[True, False, False, False, False, False, True],
        kind="stable",
    )

    assigned_volunteers = set()
    assignments = []
    for task_id in sorted(candidates["Task ID"].unique()):
        task_candidates = eligible[eligible["Task ID"] == task_id]
        for _, candidate in task_candidates.iterrows():
            volunteer_id = int(candidate["Volunteer ID"])
            if volunteer_id not in assigned_volunteers:
                assigned_volunteers.add(volunteer_id)
                item = candidate.to_dict()
                item["Explanation"] = "Eligible; selected by greedy rule-based matching"
                assignments.append(item)
                break

    if not assignments:
        return pd.DataFrame(columns=candidates.columns)
    return pd.DataFrame(assignments).sort_values("Task ID").reset_index(drop=True)

# Confirm that the weights total to 100%
def main():
    if not np.isclose(sum(WEIGHTS.values()), 1.0):
        raise ValueError(f"Weights must total 1.00, got {sum(WEIGHTS.values()):.2f}")

    volunteers, tasks = load_and_validate_data()
    candidates = build_candidate_table(volunteers, tasks)
    weighted_assignments = assign_hungarian(candidates)
    rule_assignments = assign_rule_based_greedy(candidates)

    assigned_task_ids = set(weighted_assignments["Task ID"].tolist())
    unfilled = tasks[~tasks["Task ID"].isin(assigned_task_ids)][
        ["Task ID", "Task Description", "Task Category", "Schedule"]
    ].copy()
    eligible_counts = candidates[candidates["Eligible"]].groupby("Task ID").size()
    unfilled["Reason"] = unfilled["Task ID"].apply(
        lambda task_id: (
            "No eligible volunteer available"
            if eligible_counts.get(task_id, 0) == 0
            else "Eligible volunteers were assigned to other tasks"
        )
    )

    candidates.to_csv(OUTPUT_DIR / "ranked_candidates.csv", index=False)
    weighted_assignments.to_csv(OUTPUT_DIR / "weighted_assignments.csv", index=False)
    rule_assignments.to_csv(OUTPUT_DIR / "rule_based_assignments.csv", index=False)
    unfilled.to_csv(OUTPUT_DIR / "unfilled_tasks.csv", index=False)

    print("\n=================================")
    print("EVENT VOLUNTEER ASSIGNMENT RESULTS")
    print("=================================")
    print(f"Volunteers loaded: {len(volunteers)}")
    print(f"Tasks loaded: {len(tasks)}")
    print(f"Possible volunteer-task pairs: {len(candidates)}")
    print(f"Eligible volunteer-task pairs: {int(candidates['Eligible'].sum())}")
    print(f"Hungarian assignments: {len(weighted_assignments)} / {len(tasks)}")
    print(f"Greedy rule-based assignments: {len(rule_assignments)} / {len(tasks)}")
    print(f"Unfilled tasks (Hungarian): {len(unfilled)}")

    print("\nHUNGARIAN WEIGHTED ASSIGNMENTS")
    if weighted_assignments.empty:
        print("No assignments could be made.")
    else:
        print(weighted_assignments[[
            "Task ID", "Task Description", "Volunteer ID", "Volunteer Name",
            "Distance from Event (km)", "Distance Score", "Weighted Score"
        ]].to_string(index=False))

    print("\nOutput files saved in:", OUTPUT_DIR)
    for name in ["ranked_candidates.csv", "weighted_assignments.csv",
                 "rule_based_assignments.csv", "unfilled_tasks.csv"]:
        print(" -", name)

if __name__ == "__main__":
    main()