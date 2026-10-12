# Event Volunteer Assignment System

**Group 17**

A Python-based system that ranks volunteer-task matches and assigns volunteers to event tasks using weighted scoring and assignment optimization. It also includes a greedy rule-based method as a comparison baseline

## Group Members and Roles

| Member | Role |
|---|---|
| Rafael Castillo | KNIME Workflow Developer |
| James Diaz | Data Analyst |
| Kelzo Lumabas | Python Developer |
| Benedict Roxas | Streamlit Developer |

## Features

- Loads volunteer and task information from CSV files
- Validates key fields, numeric values, unique IDs, and time ranges
- Scores every possible volunteer-task pair using five weighted criteria
- Uses the **Hungarian algorithm** to maximize the number of eligible assignments and then maximize the combined weighted score
- Runs a **greedy rule-based matching** approach for comparison
- Exports ranked candidates, assignments, and unfilled tasks as CSV files

## How the Scoring Works

| Criterion | Weight | Scoring |
|---|---:|---|
| Task category match | 35% | 100 points for a matching volunteer type and task category; otherwise 0 |
| Proficiency | 25% | Beginner: 40; Intermediate: 60; Advanced: 80; Expert: 100 |
| Availability | 20% | The volunteer must cover the entire task schedule and have enough total available hours. Eligible volunteers receive a score based on available hours relative to required task hours, capped at 100 |
| Speed | 10% | Slow: 40; Moderate: 70; Fast: 100 |
| Distance | 10% | Volunteers closer to the event receive a higher score |

The weights are configured in `WEIGHTS` and must add up to `1.00` (100%)

### Distance Scoring

The script uses the volunteer dataset's `Distance from Event (km)` column. The maximum scoring distance is `MAX_DISTANCE_KM = 5.0`

`Distance Score = max(0, 100 × (1 − distance / 5))`

Examples: 0 km = 100 points; 1 km = 80 points; 2.5 km = 50 points; 5 km or more = 0 points

The current model assumes all tasks occur at one event location. The task dataset's `Distance to Task (km)` is retained in candidate output, but the scoring function uses `Distance from Event (km)`

## Assignment Algorithms

### Primary: Weighted Scoring with Hungarian Assignment Optimization

The program creates a score matrix for volunteers and tasks. Ineligible volunteer-task pairs cannot be selected as valid assignments, and dummy columns allow tasks to remain unfilled. The algorithm prioritizes:
1. Maximizing the number of eligible assignments
2. Maximizing the combined weighted score among those assignments

Each volunteer can be assigned to at most one task, and each task can receive at most one volunteer

### Comparison: Greedy Rule-based Matching

Tasks are processed in Task ID order. For each task, the algorithm selects the best remaining eligible volunteer according to:
1. Category match
2. Proficiency
3. Availability score
4. Speed score
5. Distance score
6. Volunteer ID as a tie-breaker

This method is a baseline for comparison

## Project Structure

```text
Event Volunteer Assignment System/
├── EventVolunteerAssignment.py
├── datasets/
│   ├── EventVolunteerList - Sheet1.csv
│   └── TaskList - Sheet1.csv
└── outputs/                 # Created automatically
    ├── ranked_candidates.csv
    ├── weighted_assignments.csv
    ├── rule_based_assignments.csv
    └── unfilled_tasks.csv
```

The volunteer CSV must contain the `Distance from Event (km)` column

## Requirements and Installation

Requirements: Python, pandas, numpy, and scipy

Install dependencies in PowerShell:

```powershell
py -m pip install pandas numpy scipy
```

Use the same Python launcher to run the script, so it uses the interpreter where the packages were installed

## How to Run

1. Place `EventVolunteerAssignment.py` in the project folder
2. Put both required CSV files in the `datasets` folder
3. Open a terminal in the project folder
4. Install the dependencies if needed
5. Run:

   ```powershell
   py EventVolunteerAssignment.py
   ```

6. Review the generated files in `outputs`

## Input Data

### Volunteer Dataset

Expected columns:
- `Volunteer ID`
- `Name`
- `Skills`
- `Type`
- `Proficiency`
- `Speed`
- `Available Hours`
- `Distance from Event (km)`

`Available Hours` should be a time range such as `8:00 AM - 12:00 PM`

### Task Dataset

Expected columns:
- `Task ID`
- `Task Description`
- `Task Category`
- `Required Time (hrs)`
- `Schedule`
- `Distance to Task (km)`

`Schedule` should be a time range such as `8:00 AM - 11:00 AM`. End times must be later than start times, and required task hours must be greater than zero

## Output Files

| File | Description |
|---|---|
| `ranked_candidates.csv` | All volunteer-task pairs, scores, eligibility, and explanations |
| `weighted_assignments.csv` | Assignments produced by Hungarian optimization |
| `rule_based_assignments.csv` | Assignments produced by greedy rule-based matching |
| `unfilled_tasks.csv` | Tasks left unassigned by the Hungarian algorithm and the reason recorded by the script |

The `outputs` folder is created automatically

## Important Notes and Limitations

- **All dataset values are synthetic demonstration data**, not verified real-world volunteer information or distances
- The weights and scoring mappings are demonstration choices and can be adjusted
- Eligibility is determined by schedule coverage and sufficient available hours. A category mismatch lowers the score but does not automatically make a volunteer ineligible
- Each volunteer receives at most one task, and each task receives at most one volunteer
- Not every task is guaranteed to be filled. This can happen when there are not enough eligible volunteers or the eligible volunteers are needed for other tasks
- The distance score assumes one event location. If tasks occur at different locations, the distance model should be extended to calculate volunteer-to-task distances
- The rule-based method is greedy and does not globally optimize all assignments
- Validate the system with realistic requirements and data before using it for actual event staffing

## Troubleshooting

### CSV file not found

Check that the file names match and that both CSVs are inside `datasets`, beside the Python script's project folder

### Missing column error

Check the CSV headers against the expected columns above. In particular, the volunteer CSV must include `Distance from Event (km)`

### Time range parsing error

Use a format such as `8:00 AM - 12:00 PM`. The end time must be later than the start time. Overnight time ranges are not supported by the current parser

## Versions

v1.0.0 - Event Volunteer Assignment System Release