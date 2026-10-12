import json
import os

SUBMISSION_TASKS = {"task1": "task2_1", "task2": "task2_2", "task3": "task2_3"}


def apply_hierarchy(predictions):
    """Memes predicted as not sexist get NO in the source-intention and categorization tasks."""
    predictions = predictions.copy()
    not_sexist = predictions["task1"] == "NO"
    predictions.loc[not_sexist, "task2"] = "NO"
    predictions.loc[not_sexist, "task3"] = predictions.loc[not_sexist, "task3"].apply(lambda _: ["NO"])
    return predictions


def write_submission(predictions, out_dir=".", team="GuiGil", run=1):
    """Writes one EXIST 2026 JSON file per memes subtask (hard labels)."""
    os.makedirs(out_dir, exist_ok=True)
    for task, exist_name in SUBMISSION_TASKS.items():
        records = [{"test_case": "EXIST2026", "id": str(row["id_EXIST"]), "value": row[task]}
                   for _, row in predictions.iterrows()]
        path = os.path.join(out_dir, f"{exist_name}_hard_{team}_{run}.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(records, f, indent=4, ensure_ascii=False)
        print(f"Saved {path} ({len(records)} memes)")
