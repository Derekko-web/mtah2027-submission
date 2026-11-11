"""Player-set inputs for learning ball contact and motion at 5 Hz."""

import numpy as np


PLAYERS = 10
CHANNELS = 13
SCALE = np.array([52.5, 34], dtype=np.float32)


def ball_anchor(ball, frames):
    seen = np.isfinite(ball).all(axis=1)
    time = np.arange(len(ball))
    previous = np.maximum.accumulate(np.where(seen, time, -1000))[frames]
    following = np.minimum.accumulate(np.where(seen, time, len(ball) + 1000)[::-1])[::-1][frames]
    before, after = frames - previous, following - frames
    use = np.where(before <= after, previous, following)
    close = np.minimum(before, after) <= 15
    points = np.full((len(frames), 2), np.nan, dtype=np.float32)
    points[close] = ball[use[close]]
    return points


def velocity(positions, frames, step):
    past = positions[np.maximum(frames - step, 0)]
    present = positions[frames]
    vector = (present - past) * (25 / step)
    valid = (frames[:, None] >= step) & np.isfinite(vector).all(axis=2) & (np.linalg.norm(vector, axis=2) <= 12)
    return np.where(valid[..., None], vector, 0), valid


def features(positions, ball, teams, frames):
    positions = np.asarray(positions, dtype=np.float32)
    ball = np.asarray(ball, dtype=np.float32)
    frames = np.asarray(frames)
    anchor = ball_anchor(ball, frames)
    anchor_valid = np.isfinite(anchor).all(axis=1)
    ball_now = np.isfinite(ball[frames]).all(axis=1)
    xy = positions[frames]
    observed = np.isfinite(xy).all(axis=2)
    fast, fast_valid = velocity(positions, frames, 5)
    slow, slow_valid = velocity(positions, frames, 25)
    past_ball = ball[np.maximum(frames - 5, 0)]
    ball_velocity = (ball[frames] - past_ball) * 5
    ball_velocity_ok = ((frames >= 5) & np.isfinite(ball_velocity).all(axis=1)
                        & (np.linalg.norm(ball_velocity, axis=1) < 45))
    ball_velocity = np.where(ball_velocity_ok[:, None], ball_velocity, 0)

    output = np.zeros((len(frames), 2, PLAYERS, CHANNELS), dtype=np.float16)
    for team in (0, 1):
        side = teams == team
        here, present = xy[:, side], observed[:, side]
        if not side.any():
            continue
        centroid = np.where(present[..., None], here, 0).sum(axis=1) / np.maximum(1, present.sum(axis=1, keepdims=True))
        relative = anchor[:, None] - here
        radius = np.linalg.norm(relative, axis=2)
        team_radius = np.linalg.norm(here - centroid[:, None], axis=2)
        priority = np.where(anchor_valid[:, None], radius, team_radius)
        order = np.argsort(np.where(present, priority, np.inf), axis=1, kind="stable")[:, :PLAYERS]
        radial_speed = (fast[:, side] * relative).sum(axis=2) / np.maximum(radius, .25)
        relative_motion = np.linalg.norm(fast[:, side] - ball_velocity[:, None], axis=2)
        relative_motion = np.where(ball_velocity_ok[:, None] & fast_valid[:, side], relative_motion, 0)
        fields = np.stack((np.abs(here[..., 0]) / SCALE[0], np.abs(here[..., 1]) / SCALE[1],
                           np.abs(relative[..., 0]) / 20, np.abs(relative[..., 1]) / 20,
                           np.log1p(radius) / 4, radial_speed / 8, relative_motion / 25,
                           np.linalg.norm(fast[:, side], axis=2) / 8, np.linalg.norm(slow[:, side], axis=2) / 8,
                           team_radius / 25, fast_valid[:, side], slow_valid[:, side], present), axis=-1)
        fields = np.where(present[..., None], fields, 0)
        fields[..., 2:6] *= np.where(ball_now, 1., .75)[:, None, None]
        selected = np.take_along_axis(fields, order[..., None], axis=1)
        output[:, team, :order.shape[1]] = np.clip(np.nan_to_num(selected), -4, 4).astype(np.float16)
    return output
