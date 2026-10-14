# Group 17 - Event Volunteer Assignment System (Streamlit app)
# Run with: py -m streamlit run app.py

from pathlib import Path
import tempfile
import numpy as np
import pandas as pd
import streamlit as st
import EventVolunteerAssignment as eva  # your ORIGINAL script, unchanged

st.set_page_config(page_title="Event Volunteer Assignment", layout="wide")

DATA_DIR = Path(__file__).resolve().parent / "datasets"
OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"   # created automatically if missing
DEFAULT_VOL = Path(eva.VOLUNTEER_FILE)
DEFAULT_TASK = Path(eva.TASK_FILE)
DEFAULT_WEIGHTS = dict(eva.WEIGHTS)          # copy of the original defaults
DEFAULT_MAX_KM = float(eva.MAX_DISTANCE_KM)


def run_original_pipeline(vol_path, task_path, weights, max_km):
    """Drives the original script by setting its globals, then calling its own functions."""
    eva.VOLUNTEER_FILE = Path(vol_path)
    eva.TASK_FILE = Path(task_path)
    eva.WEIGHTS.update(weights)               # score_pair reads this global
    eva.MAX_DISTANCE_KM = float(max_km)       # and this one
    if not np.isclose(sum(eva.WEIGHTS.values()), 1.0):
        raise ValueError("Weights must total 1.00")

    volunteers, tasks = eva.load_and_validate_data()
    candidates = eva.build_candidate_table(volunteers, tasks)
    weighted = eva.assign_hungarian(candidates)
    rule_based = eva.assign_rule_based_greedy(candidates)

    # Same unfilled-task logic as main() in the original script
    assigned = set(weighted["Task ID"].tolist())
    unfilled = tasks[~tasks["Task ID"].isin(assigned)][
        ["Task ID", "Task Description", "Task Category", "Schedule"]].copy()
    counts = candidates[candidates["Eligible"]].groupby("Task ID").size()
    unfilled["Reason"] = unfilled["Task ID"].apply(
        lambda t: "No eligible volunteer available" if counts.get(t, 0) == 0
        else "Eligible volunteers were assigned to other tasks")
    return {"volunteers": volunteers, "tasks": tasks, "candidates": candidates,
            "weighted": weighted, "rule_based": rule_based,
            "unfilled": unfilled.reset_index(drop=True)}


def save_upload(uploaded, folder):
    path = Path(folder) / uploaded.name
    path.write_bytes(uploaded.getvalue())
    return path

WEIGHT_LABELS = {
    "category": "Task category match",
    "proficiency": "Proficiency",
    "availability": "Availability",
    "speed": "Speed",
    "distance": "Distance",
}

SHOW_COLS = ["Task ID", "Task Description", "Volunteer ID", "Volunteer Name",
             "Volunteer Category", "Proficiency", "Speed", "Distance from Event (km)",
             "Weighted Score"]


def save_to_outputs(df, filename):
    """Write a CSV into the project's outputs folder, creating the folder if needed."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_DIR / filename, index=False)
    st.toast(f"Saved {filename} to the outputs folder")


def csv_bytes(df):
    return df.to_csv(index=False).encode("utf-8")


def summarize(assignments, total_tasks):
    scores = assignments["Weighted Score"].astype(float) if not assignments.empty else pd.Series(dtype=float)
    return {
        "filled": len(assignments),
        "rate": len(assignments) / total_tasks * 100 if total_tasks else 0,
        "total": scores.sum(),
        "avg": scores.mean() if len(scores) else 0.0,
        "cat": int(assignments["Category Match"].sum()) if not assignments.empty else 0,
    }


# ---------------------------------------------------------------- header
st.title("Event Volunteer Assignment System")
st.caption("Group 17 · Weighted scoring + Hungarian optimization, compared against a greedy rule-based baseline. "
           "All data is synthetic demonstration data.")

# ---------------------------------------------------------------- sidebar
with st.sidebar:
    st.header("1. Data")
    vol_file = st.file_uploader("Volunteer CSV", type="csv")
    task_file = st.file_uploader("Task CSV", type="csv")
    has_defaults = DEFAULT_VOL.exists() and DEFAULT_TASK.exists()
    if not (vol_file and task_file):
        if has_defaults:
            st.info("No upload yet, so the sample files in `datasets/` will be used.")
        else:
            st.warning("Upload both CSV files to continue.")

    st.header("2. Scoring weights (%)")
    weights_pct = {}
    for key, label in WEIGHT_LABELS.items():
        weights_pct[key] = st.slider(label, 0, 100, int(round(DEFAULT_WEIGHTS[key] * 100)), step=5)
    total_pct = sum(weights_pct.values())
    if total_pct == 100:
        st.success("Total: 100%")
    else:
        st.error(f"Total: {total_pct}%. Weights must add up to 100%.")

    max_km = st.number_input("Max scoring distance (km)", min_value=0.5, value=DEFAULT_MAX_KM, step=0.5,
                             help="Volunteers at or beyond this distance get 0 distance points.")

    can_run = total_pct == 100 and (bool(vol_file and task_file) or has_defaults)
    run = st.button("Run assignment", type="primary", width="stretch", disabled=not can_run)

# ---------------------------------------------------------------- run
if run:
    try:
        weights = {k: v / 100 for k, v in weights_pct.items()}
        with tempfile.TemporaryDirectory() as tmp:
            vol_path = save_upload(vol_file, tmp) if vol_file else DEFAULT_VOL
            task_path = save_upload(task_file, tmp) if task_file else DEFAULT_TASK
            st.session_state["result"] = run_original_pipeline(vol_path, task_path, weights, max_km)
        st.session_state["settings"] = {"weights": weights_pct, "max_km": max_km}
    except Exception as e:
        st.session_state.pop("result", None)
        st.error(f"Could not run the assignment: {e}")

if "result" not in st.session_state:
    st.markdown("### How to use")
    st.markdown(
        "1. Upload the volunteer and task CSVs, or use the sample data.\n"
        "2. Adjust the scoring weights (they must total 100%).\n"
        "3. Press **Run assignment** in the sidebar."
    )
    with st.expander("How scoring works"):
        st.markdown(
            "Every volunteer-task pair gets a weighted score from five criteria: category match, proficiency, "
            "availability, speed, and distance. A volunteer is **eligible** only if they are available for the whole "
            "task schedule and have enough hours. The **Hungarian algorithm** then picks the set of assignments that "
            "fills the most tasks and, among those, has the highest total score. The **greedy** method goes task by "
            "task and grabs the best remaining volunteer, which can block a better overall match."
        )
    st.stop()

res = st.session_state["result"]
volunteers, tasks = res["volunteers"], res["tasks"]
candidates, weighted, rule_based, unfilled = res["candidates"], res["weighted"], res["rule_based"], res["unfilled"]
n_tasks = len(tasks)

# ---------------------------------------------------------------- overview
m1, m2, m3, m4 = st.columns(4)
m1.metric("Volunteers", len(volunteers))
m2.metric("Tasks", n_tasks)
m3.metric("Eligible pairs", f"{int(candidates['Eligible'].sum())} / {len(candidates)}")
m4.metric("Unfilled (Hungarian)", len(unfilled))

tab_cmp, tab_hung, tab_greedy, tab_unfilled, tab_cand = st.tabs(
    ["Comparison", "Hungarian assignments", "Greedy assignments", "Unfilled tasks", "Ranked candidates"]
)

# ---------------------------------------------------------------- comparison
with tab_cmp:
    sh, sg = summarize(weighted, n_tasks), summarize(rule_based, n_tasks)

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Hungarian (weighted)")
        a, b = st.columns(2)
        a.metric("Tasks filled", f"{sh['filled']} / {n_tasks}")
        b.metric("Total score", f"{sh['total']:.2f}", delta=f"{sh['total'] - sg['total']:+.2f} vs greedy")
        a, b = st.columns(2)
        a.metric("Average score", f"{sh['avg']:.2f}")
        b.metric("Category matches", sh["cat"])
    with c2:
        st.subheader("Greedy (rule-based)")
        a, b = st.columns(2)
        a.metric("Tasks filled", f"{sg['filled']} / {n_tasks}")
        b.metric("Total score", f"{sg['total']:.2f}")
        a, b = st.columns(2)
        a.metric("Average score", f"{sg['avg']:.2f}")
        b.metric("Category matches", sg["cat"])

    if sh["filled"] > sg["filled"]:
        st.success(f"Hungarian filled {sh['filled'] - sg['filled']} more task(s) than greedy.")
    elif sh["total"] > sg["total"] + 1e-9:
        st.success(f"Both fill {sh['filled']} tasks, but Hungarian's total score is {sh['total'] - sg['total']:.2f} higher.")
    else:
        st.info("On this data and these weights, both methods perform the same.")

    st.subheader("Per-task comparison")
    h = weighted[["Task ID", "Volunteer Name", "Weighted Score"]].rename(
        columns={"Volunteer Name": "Hungarian volunteer", "Weighted Score": "Hungarian score"})
    g = rule_based[["Task ID", "Volunteer Name", "Weighted Score"]].rename(
        columns={"Volunteer Name": "Greedy volunteer", "Weighted Score": "Greedy score"})
    cmp_df = tasks[["Task ID", "Task Description"]].merge(h, on="Task ID", how="left").merge(g, on="Task ID", how="left")
    cmp_df["Same pick?"] = cmp_df["Hungarian volunteer"].fillna("-") == cmp_df["Greedy volunteer"].fillna("-")
    only_diff = st.checkbox("Show only tasks where the methods differ")
    st.dataframe(cmp_df[~cmp_df["Same pick?"]] if only_diff else cmp_df, width="stretch", hide_index=True)

    if not weighted.empty or not rule_based.empty:
        st.subheader("Score per assigned task")
        chart = pd.DataFrame({"Task ID": tasks["Task ID"]}).merge(
            weighted[["Task ID", "Weighted Score"]].rename(columns={"Weighted Score": "Hungarian"}), on="Task ID", how="left"
        ).merge(
            rule_based[["Task ID", "Weighted Score"]].rename(columns={"Weighted Score": "Greedy"}), on="Task ID", how="left"
        ).fillna(0).set_index("Task ID")
        st.bar_chart(chart)

# ---------------------------------------------------------------- assignment tabs
def assignment_tab(df, filename, key):
    if df.empty:
        st.warning("No assignments could be made.")
        return
    st.dataframe(df[SHOW_COLS], width="stretch", hide_index=True)
    st.download_button("Download CSV", csv_bytes(df), filename, "text/csv", key=key,
                       on_click=save_to_outputs, args=(df, filename))

with tab_hung:
    assignment_tab(weighted, "weighted_assignments.csv", "dl_h")

with tab_greedy:
    assignment_tab(rule_based, "rule_based_assignments.csv", "dl_g")

with tab_unfilled:
    if unfilled.empty:
        st.success("Every task was filled.")
    else:
        st.dataframe(unfilled, width="stretch", hide_index=True)
        st.download_button("Download CSV", csv_bytes(unfilled), "unfilled_tasks.csv", "text/csv", key="dl_u",
                           on_click=save_to_outputs, args=(unfilled, "unfilled_tasks.csv"))

# ---------------------------------------------------------------- candidates
with tab_cand:
    f1, f2, f3 = st.columns([2, 1, 1])
    task_choice = f1.selectbox(
        "Task", ["All tasks"] + [f"{r['Task ID']} - {r['Task Description']}" for _, r in tasks.iterrows()])
    only_eligible = f2.checkbox("Eligible only", value=True)
    top_n = f3.number_input("Top N per task (0 = all)", min_value=0, value=0, step=1)

    view = candidates
    if task_choice != "All tasks":
        view = view[view["Task ID"] == int(task_choice.split(" - ")[0])]
    if only_eligible:
        view = view[view["Eligible"]]
    if top_n > 0:
        view = view.groupby("Task ID", sort=False).head(int(top_n))

    st.caption(f"{len(view)} row(s) shown")
    st.dataframe(view, width="stretch", hide_index=True)
    st.download_button("Download full ranked candidates CSV", csv_bytes(candidates), "ranked_candidates.csv",
                       "text/csv", key="dl_c", on_click=save_to_outputs, args=(candidates, "ranked_candidates.csv"))
