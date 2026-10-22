"""Extra stoppage examples from Metrica's public, pre-2024 sample matches.

Source: https://github.com/metrica-sports/sample-data, pinned below. These are
event-derived weak training labels from different games, not competition ground
truth. Data is downloaded into ignored data/ and is never committed or uploaded.
"""

import concurrent.futures
import hashlib
import json
from pathlib import Path
import sys
import urllib.request

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from games import GAME, GAMES, require
from starter.baseline import team_of_each_track

COMMIT = "e706dd506b360d69d9d123d5b8026e7294b13996"
DATA = ROOT/"data/external_metrica"


def download():
    DATA.mkdir(exist_ok=True)
    paths = [f"data/Sample_Game_{game}/Sample_Game_{game}_Raw{suffix}.csv"
             for game in (1, 2) for suffix in
             ("EventsData", "TrackingData_Home_Team", "TrackingData_Away_Team")]
    def get(name):
        url = f"https://raw.githubusercontent.com/metrica-sports/sample-data/{COMMIT}/{name}"
        target = DATA/Path(name).name
        if not target.exists():
            with urllib.request.urlopen(url, timeout=300) as response:
                content = response.read()
            temporary = target.with_suffix(".partial")
            temporary.write_bytes(content)
            temporary.replace(target)
        return dict(url=url, file=target.name, sha256=hashlib.sha256(target.read_bytes()).hexdigest())
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        files = list(pool.map(get, paths))
    (DATA/"source_manifest.json").write_text(json.dumps(dict(
        source="Metrica Sports public sample data", commit=COMMIT, files=files,
        attribution="Metrica Sports, https://github.com/metrica-sports/sample-data",
        limitations="Event-derived weak labels; different fixtures, not official competition targets."), indent=2)+"\n")


def weak_labels(events, frames, periods):
    labels = np.full(len(frames), "X", dtype="<U1")
    live_types = {"PASS", "SHOT", "RECOVERY", "CHALLENGE", "BALL LOST"}
    live_events = events[events.Type.isin(live_types)]
    for row in live_events.to_dict("records"):
        start, stop = int(row["Start Frame"]), int(row["End Frame"])
        if stop < start:
            continue
        left, right = np.searchsorted(frames, [start, stop+1])
        labels[left:right] = "L"
    terminal = events[(events.Type.isin(["BALL OUT", "FAULT RECEIVED"])) |
                      ((events.Type == "SHOT") & events.Subtype.fillna("").str.contains("GOAL")) |
                      ((events.Type == "BALL LOST") & events.Subtype.fillna("").str.contains("OFFSIDE"))]
    spells = 0
    for row in events[events.Type == "SET PIECE"].to_dict("records"):
        kick, period = int(row["Start Frame"]), row["Period"]
        candidates = terminal[(terminal.Period == period) & (terminal["End Frame"] < kick)]
        if candidates.empty:
            continue
        stop = int(candidates["End Frame"].max())
        if not 25 <= kick-stop <= 75*25:
            continue
        # Reject stale terminations if meaningful play was tagged afterwards.
        intervening = live_events[(live_events.Period == period) &
                                  (live_events["Start Frame"] > stop+12) &
                                  (live_events["Start Frame"] < kick-12) &
                                  live_events.Type.isin(["PASS", "SHOT", "CHALLENGE"])]
        if not intervening.empty:
            continue
        left, right = np.searchsorted(frames, [stop+13, kick-12])
        labels[left:right] = np.where(periods[left:right] == period, "D", "X")
        spells += 1
    return labels, spells


def feature_arrays(positions, speeds, ball, teams, pitch):
    n_frames = len(ball)
    values = {}
    valid = np.isfinite(ball).all(axis=1)
    values["ball_seen"] = valid.astype(float)
    values["ball_touchline_distance"] = np.minimum(ball[:, 1], pitch[1]-ball[:, 1])
    values["ball_goal_line_distance"] = np.minimum(ball[:, 0], pitch[0]-ball[:, 0])
    for step in (5, 25):
        displacement = np.full(n_frames, np.nan)
        displacement[step:] = np.linalg.norm(ball[step:]-ball[:-step], axis=1)*25/step
        values[f"ball_displacement_{step}"] = displacement
    paired = {}
    for team in (0, 1):
        xy, speed = positions[:, teams == team], speeds[:, teams == team]
        speed = np.where(np.isfinite(xy).all(axis=2) & (speed >= 0) & (speed <= 15), speed, np.nan)
        observed = np.isfinite(speed)
        moving = np.where(observed, (speed > 1).astype(float), np.nan)
        with np.errstate(invalid="ignore"):
            median = pd.DataFrame(speed).median(axis=1).to_numpy()
            fraction = pd.DataFrame(moving).mean(axis=1).to_numpy()
        distance = np.linalg.norm(xy-ball[:, None, :], axis=2)
        order = np.argsort(np.where(np.isfinite(distance), distance, np.inf), axis=1)
        nearest = np.take_along_axis(distance, order[:, :1], axis=1)[:, 0]
        nearest_speed = np.take_along_axis(speed, order[:, :1], axis=1)[:, 0]
        paired[team] = dict(speed_median=median, moving_fraction=fraction,
                            nearest_distance=nearest, nearest_speed=nearest_speed)
    for name in paired[0]:
        pair = pd.DataFrame({0: paired[0][name], 1: paired[1][name]})
        values[name+"_minimum"] = pair.min(axis=1)
        values[name+"_maximum"] = pair.max(axis=1)
    base = pd.DataFrame(values)
    extra = {}
    for name in ("ball_displacement_25", "speed_median_minimum", "speed_median_maximum",
                 "nearest_distance_minimum", "moving_fraction_maximum"):
        for seconds in (1, 3):
            extra[f"{name}_mean_{seconds}s"] = base[name].rolling(seconds*25+1, center=True, min_periods=13).mean()
    return base.join(pd.DataFrame(extra)).replace([np.inf, -np.inf], np.nan).astype(np.float32)


def load_sample(game):
    home = pd.read_csv(DATA/f"Sample_Game_{game}_RawTrackingData_Home_Team.csv", skiprows=2)
    away = pd.read_csv(DATA/f"Sample_Game_{game}_RawTrackingData_Away_Team.csv", skiprows=2)
    if not np.array_equal(home.Frame, away.Frame):
        raise ValueError("External team timelines differ")
    frames = home.Frame.to_numpy()
    periods = home.Period.to_numpy()
    if not np.all(np.diff(frames) == 1):
        raise ValueError("External tracking is not contiguous")
    pitch = np.array([105., 68.])
    coordinates = [table.iloc[:, 3:-2].to_numpy(dtype=float).reshape(len(table), -1, 2)*pitch
                   for table in (home, away)]
    ball = home.iloc[:, -2:].to_numpy(dtype=float)*pitch
    positions = np.concatenate(coordinates, axis=1)
    teams = np.r_[np.zeros(coordinates[0].shape[1], dtype=int), np.ones(coordinates[1].shape[1], dtype=int)]
    speeds = np.full(positions.shape[:2], np.nan)
    speeds[5:] = np.linalg.norm(positions[5:]-positions[:-5], axis=2)/.2
    speeds[np.r_[np.zeros(5, dtype=bool), periods[5:] != periods[:-5]]] = np.nan
    events = pd.read_csv(DATA/f"Sample_Game_{game}_RawEventsData.csv")
    labels, spells = weak_labels(events, frames, periods)
    return feature_arrays(positions, speeds, ball, teams, pitch), labels, spells


def competition_features(n_frames):
    metadata, tracking = require(["atd_meta", "atd_raw"])
    teams = team_of_each_track(metadata)
    pitch = np.array(GAMES[GAME]["pitch"])
    positions = np.full((n_frames, len(teams), 2), np.nan)
    speeds = np.full((n_frames, len(teams)), np.nan)
    ball = np.full((n_frames, 2), np.nan)
    count = 0
    with open(tracking) as handle:
        for frame, line in enumerate(handle):
            if frame >= n_frames:
                break
            index, raw, raw_ball = line.strip().split(":")
            if int(index) != frame+1:
                raise ValueError("Competition tracking alignment failed")
            players = np.fromstring(raw.replace(";", ","), sep=",").reshape(-1, 3)
            positions[frame] = players[:, :2]*pitch
            speeds[frame] = players[:, 2]
            ball[frame] = np.fromstring(raw_ball, sep=",")[:2]*pitch
            count += 1
    if count != n_frames:
        raise ValueError("Competition tracking ended early")
    return feature_arrays(positions, speeds, ball, teams, pitch)


if __name__ == "__main__":
    download()
    summaries = []
    for game in (1, 2):
        x, labels, spells = load_sample(game)
        x.to_pickle(DATA/f"game_{game}_features.pkl")
        np.save(DATA/f"game_{game}_weak_labels.npy", labels)
        summaries.append(dict(game=game, derived_stoppages=spells,
                              labelled_frames=int((labels != "X").sum()),
                              dead_frames=int((labels == "D").sum()),
                              visible_dead_fraction=float(x.loc[labels == "D", "ball_seen"].mean())))
    (DATA/"weak_label_summary.json").write_text(json.dumps(summaries, indent=2)+"\n")
    print(json.dumps(summaries, indent=2))
