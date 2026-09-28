"""Live Google Flights experiment: accuracy + latency + cost, configurable future date.

Usage:
  uv run --env-file .env python experiments/flights_experiment.py --trials 3
Writes artifacts/experiments/flights/<timestamp>/results.json
and artifacts/experiments/flights/<timestamp>/trial<N>/state.json
"""

import argparse
import base64
import json
import time
from datetime import datetime
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from jev_ultrafast import Agent

URL = "https://www.google.com/travel/flights?hl=en"


def labels(date_iso):
    dt = datetime.strptime(date_iso, "%Y-%m-%d")
    short = f"{dt.strftime('%a')}, {dt.strftime('%b')} {dt.day}"  # Fri, Nov 20
    long = f"{dt.strftime('%A')}, {dt.strftime('%B')} {dt.day}"  # Friday, November 20
    spoken = f"{dt.strftime('%B')} {dt.day}, {dt.year}"  # November 20, 2026
    return short, long, spoken


def goal_for(date_iso):
    _, _, spoken = labels(date_iso)
    return (
        f"Find one-way flights from Zurich to London on {spoken}, for one adult in economy. "
        "Stop when matching flight options are visible. Do not select or book a flight."
    )


def verify(page, date_iso, date_short, date_long):
    parsed = urlparse(page["url"])
    encoded = parse_qs(parsed.query).get("tfs", [""])[0]
    try:
        date_in_url = date_iso.encode() in base64.urlsafe_b64decode(encoded + "=" * (-len(encoded) % 4))
    except ValueError:
        date_in_url = False
    actions = page["actions"]
    values = {a["label"].strip(): a.get("value") for a in actions}
    flights = [a["label"] for a in actions if "Select flight" in a["label"]]
    checks = {
        "search_page": parsed.hostname == "www.google.com" and parsed.path == "/travel/flights/search",
        "one_way": values.get("Change ticket type. One way") == "One way",
        "origin": values.get("Where from?") == "Zürich",
        "destination": values.get("Where to?") == "London",
        "date": values.get("Departure") == date_short,
        "year": date_in_url or f"departing {date_iso}" in page["text"],
        "results": bool(flights) and all(date_long in f for f in flights),
    }
    return {"passed": all(checks.values()), "checks": checks, "visible_flights": flights}


def summarize(state):
    hist = state["history"]
    lat = [h["latency_ms"] for h in hist if h.get("latency_ms") is not None]
    return {
        "actions": len(hist),
        "status": state["status"],
        "elapsed_ms": state["elapsed_ms"],
        "decision_latency_ms": {
            "n": len(lat),
            "median": sorted(lat)[len(lat) // 2] if lat else None,
            "max": max(lat) if lat else None,
        },
        "jev_input_tokens": sum((h.get("usage") or {}).get("input_tokens", 0) for h in hist),
        "jev_cost_usd": round(sum((h.get("usage") or {}).get("cost", 0) for h in hist), 6),
        "text_helper_calls": len(state.get("text_calls", [])),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=int, default=1)
    parser.add_argument("--date", default="2026-11-20")
    parser.add_argument("--root", default="artifacts/experiments/flights")
    args = parser.parse_args()

    date_short, date_long, _ = labels(args.date)
    goal = goal_for(args.date)
    run_dir = Path(args.root) / time.strftime("%Y%m%d-%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)

    tasks = []
    for trial in range(args.trials):
        trial_dir = run_dir / f"trial{trial + 1}"
        trial_dir.mkdir(parents=True, exist_ok=True)
        agent = Agent(URL, goal)
        started = time.perf_counter()
        error = None
        try:
            for state in agent.run():
                last = state["history"][-1] if state["history"] else {}
                print(
                    f"  t{trial + 1} {state['elapsed_ms']:>6} ms  {state['status']:<7} {last.get('action', '')}",
                    flush=True,
                )
        except Exception as exc:  # noqa: BLE001
            error = f"{type(exc).__name__}: {exc}"
        finally:
            wall_ms = round((time.perf_counter() - started) * 1000)
            state = agent.snapshot()
            final = agent.browser.observe(screenshot=True)
            state["verification"] = verify(final, args.date, date_short, date_long)
            task = {
                "trial": trial + 1,
                "wall_ms_incl_setup": wall_ms,
                "error": error,
                "goal": goal,
                **summarize(state),
                "verification": state["verification"],
            }
            tasks.append(task)
            (trial_dir / "state.json").write_text(json.dumps(state, indent=2))
            agent.close()

    passed = sum(1 for t in tasks if t["verification"]["passed"])
    summary = {
        "model": "typesafe/jev-1.13",
        "endpoint": "openrouter.ai System One API",
        "date": args.date,
        "trials": args.trials,
        "success": passed,
        "accuracy": round(passed / args.trials, 3),
        "median_actions": sorted(t["actions"] for t in tasks)[len(tasks) // 2],
        "median_decision_latency_ms": sorted(
            t["decision_latency_ms"]["median"] for t in tasks if t["decision_latency_ms"]["median"]
        )[max(0, len(tasks) - 1) // 2]
        if tasks
        else None,
        "total_jev_cost_usd": round(sum(t["jev_cost_usd"] for t in tasks), 6),
        "tasks": tasks,
    }
    (run_dir / "results.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps({k: v for k, v in summary.items() if k != "tasks"}, indent=2))
    print("run dir:", run_dir)


if __name__ == "__main__":
    main()
