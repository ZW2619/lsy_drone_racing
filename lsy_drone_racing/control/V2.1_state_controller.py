from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from crazyflow.sim.visualize import draw_line, draw_points
from scipy.interpolate import PchipInterpolator
from scipy.spatial.transform import Rotation as R

from lsy_drone_racing.control import Controller

if TYPE_CHECKING:
    from crazyflow import Sim
    from numpy.typing import NDArray


class V2StateController(Controller):
    """Plan one gate at a time and update the plan when true object poses become visible."""

    def __init__(self, obs: dict[str, NDArray[np.floating]], info: dict, config: dict):
        super().__init__(obs, info, config)

        self._freq = float(config.env.freq)
        self._cruise_speed = 0.42
        self._gate_offset = 0.45
        self._gate_alignment_offset = 0.12
        self._gate_recovery_offset = 0.62
        self._takeoff_height = 0.45
        self._minimum_segment_time = 0.45
        self._obstacle_clearance = 0.25
        self._avoidance_margin = 0.06
        self._tracking_slowdown_distance = 0.45
        self._exit_reached_distance = 0.18

        self._active_pass_index = int(obs["n_gates_passed"])
        self._pending_pass_index: int | None = None
        self._finished = False
        self._desired_yaw = 0.0
        self._last_desired_position = np.asarray(obs["pos"], dtype=np.float64).copy()

        self._known_gates: dict[int, tuple[NDArray[np.float64], NDArray[np.float64]]] = {}
        self._known_obstacles: dict[int, NDArray[np.float64]] = {}
        self._gate_visited = np.asarray(obs["gates_visited"], dtype=bool).copy()
        self._obstacles_visited = np.asarray(obs["obstacles_visited"], dtype=bool).copy()
        self._remember_observed_objects(obs)

        self._local_spline: PchipInterpolator | None = None
        self._local_waypoints = np.empty((0, 3), dtype=np.float64)
        self._local_times = np.empty(0, dtype=np.float64)
        self._trajectory_time = 0.0
        self._trajectory_duration = 0.0
        self._current_gate_id = -1
        self._current_exit = np.asarray(obs["pos"], dtype=np.float64).copy()
        self._planned_gate_pos = np.zeros(3)
        self._planned_gate_quat = np.array([0.0, 0.0, 0.0, 1.0])

        self._plan_current_gate(obs, include_takeoff=True)

    def _remember_observed_objects(self, obs: dict[str, NDArray[np.floating]]) -> None:
        """Cache true poses once objects enter sensor range."""
        gate_visited = np.asarray(obs["gates_visited"], dtype=bool)
        for gate_id in np.flatnonzero(gate_visited):
            self._known_gates[int(gate_id)] = (
                np.asarray(obs["gates_pos"][gate_id], dtype=np.float64).copy(),
                np.asarray(obs["gates_quat"][gate_id], dtype=np.float64).copy(),
            )

        obstacles_visited = np.asarray(obs["obstacles_visited"], dtype=bool)
        for obstacle_id in np.flatnonzero(obstacles_visited):
            self._known_obstacles[int(obstacle_id)] = np.asarray(
                obs["obstacles_pos"][obstacle_id], dtype=np.float64
            ).copy()

    def _current_gate_geometry(
        self, obs: dict[str, NDArray[np.floating]]
    ) -> tuple[int, float, NDArray[np.float64], NDArray[np.float64]]:
        """Return gate ID, direction and the best currently available gate pose."""
        gate_id = int(obs["gate_sequence"][self._active_pass_index])
        direction = float(obs["gate_sequence_direction"][self._active_pass_index])
        if gate_id in self._known_gates:
            center, quaternion = self._known_gates[gate_id]
        else:
            center = np.asarray(obs["gates_pos"][gate_id], dtype=np.float64)
            quaternion = np.asarray(obs["gates_quat"][gate_id], dtype=np.float64)
        return gate_id, direction, center.copy(), quaternion.copy()

    def _available_obstacles(
        self, obs: dict[str, NDArray[np.floating]]
    ) -> NDArray[np.float64]:
        """Use true obstacle positions when known and nominal positions otherwise."""
        obstacles = np.asarray(obs["obstacles_pos"], dtype=np.float64).copy()
        for obstacle_id, position in self._known_obstacles.items():
            obstacles[obstacle_id] = position
        return obstacles

    def _plan_current_gate(
        self, obs: dict[str, NDArray[np.floating]], include_takeoff: bool = False
    ) -> None:
        """Build a short trajectory from the current state through the active gate."""
        gate_id, direction, center, quaternion = self._current_gate_geometry(obs)
        normal = R.from_quat(quaternion).apply([1.0, 0.0, 0.0]) * direction
        obstacles = self._available_obstacles(obs)

        current_position = np.asarray(obs["pos"], dtype=np.float64)
        approach = center - self._gate_offset * normal
        exit_point = center + self._gate_offset * normal
        approach = self._move_staging_point(approach, normal, obstacles)
        exit_point = self._move_staging_point(exit_point, normal, obstacles)
        alignment_entry = center - self._gate_alignment_offset * normal
        alignment_exit = center + self._gate_alignment_offset * normal

        waypoints = [current_position.copy()]
        if include_takeoff and current_position[2] < self._takeoff_height - 0.05:
            takeoff = current_position.copy()
            takeoff[2] = self._takeoff_height
            waypoints.append(takeoff)

        # A newly observed gate may move while the drone is already near its nominal approach.
        # Do not command a return to a staging point that is already behind the drone.
        local_x = float(np.dot(current_position - center, normal))
        if local_x >= 0.0:
            # The gate has not been counted yet, but the drone is already on its exit side.  This
            # can happen when the true pose is revealed late and differs from the nominal pose.
            # Never command the center from here, because that would cross the gate backwards.
            # Return to the entry side outside the physical frame first.
            waypoints.extend(
                self._build_entry_side_recovery(
                    current_position, center, normal, obstacles
                )
            )
            if np.linalg.norm(waypoints[-1] - approach) > 0.12:
                waypoints.append(approach)
            waypoints.append(alignment_entry)
        elif local_x < -self._gate_alignment_offset:
            if np.linalg.norm(current_position - approach) > 0.12:
                waypoints.extend(self._route_to_approach(current_position, approach, obstacles))
            waypoints.append(alignment_entry)
        elif local_x < 0.0:
            waypoints.append(alignment_entry)

        waypoints.extend((center, alignment_exit, exit_point))
        self._local_waypoints = self._remove_duplicate_waypoints(waypoints)
        self._local_times = self._allocate_waypoint_times(self._local_waypoints)
        self._trajectory_duration = float(self._local_times[-1])
        self._trajectory_time = 0.0
        self._local_spline = PchipInterpolator(
            self._local_times, self._local_waypoints, axis=0
        )

        self._current_gate_id = gate_id
        self._current_exit = exit_point
        self._planned_gate_pos = center
        self._planned_gate_quat = quaternion
        self._last_desired_position = current_position.copy()

    def _build_entry_side_recovery(
        self,
        start: NDArray[np.float64],
        center: NDArray[np.float64],
        normal: NDArray[np.float64],
        obstacles: NDArray[np.float64],
    ) -> list[NDArray[np.float64]]:
        """Return from a gate's exit side to its entry side outside the gate frame."""
        lateral = np.array([-normal[1], normal[0], 0.0], dtype=np.float64)
        lateral /= max(np.linalg.norm(lateral), 1e-9)

        candidates = []
        for sign in (1.0, -1.0):
            side_offset = sign * self._gate_recovery_offset * lateral
            outside_exit = center + self._gate_offset * normal + side_offset
            outside_entry = center - self._gate_offset * normal + side_offset
            path_length = np.linalg.norm(outside_exit - start) + np.linalg.norm(
                outside_entry - outside_exit
            )
            if len(obstacles):
                clearance = min(
                    np.min(np.linalg.norm(obstacles[:, :2] - outside_exit[:2], axis=1)),
                    np.min(np.linalg.norm(obstacles[:, :2] - outside_entry[:2], axis=1)),
                )
            else:
                clearance = np.inf
            candidates.append((clearance, path_length, outside_exit, outside_entry))

        safe_candidates = [
            candidate
            for candidate in candidates
            if candidate[0] >= self._obstacle_clearance
        ]
        selected = min(safe_candidates or candidates, key=lambda item: item[1])
        return [selected[2], selected[3]]

    def _route_to_approach(
        self,
        start: NDArray[np.float64],
        approach: NDArray[np.float64],
        obstacles: NDArray[np.float64],
    ) -> list[NDArray[np.float64]]:
        """Insert one conservative side point if the direct approach crosses an obstacle buffer."""
        if not len(obstacles):
            return [approach]

        segment = approach[:2] - start[:2]
        length_squared = float(segment @ segment)
        if length_squared < 1e-10:
            return [approach]

        required_clearance = self._obstacle_clearance + self._avoidance_margin
        conflicts = []
        for obstacle in obstacles:
            parameter = float(
                np.clip((obstacle[:2] - start[:2]) @ segment / length_squared, 0.0, 1.0)
            )
            closest = start[:2] + parameter * segment
            distance = np.linalg.norm(obstacle[:2] - closest)
            if distance < required_clearance and 0.05 < parameter < 0.95:
                conflicts.append((parameter, obstacle))

        if not conflicts:
            return [approach]

        _, obstacle = min(conflicts, key=lambda item: item[0])
        direction = segment / np.linalg.norm(segment)
        lateral = np.array([-direction[1], direction[0]])
        height = 0.5 * (start[2] + approach[2])
        candidates = []
        for sign in (1.0, -1.0):
            candidate = np.array(
                [
                    *(obstacle[:2] + sign * required_clearance * lateral),
                    height,
                ]
            )
            path_length = np.linalg.norm(candidate - start) + np.linalg.norm(approach - candidate)
            min_clearance = np.min(
                np.linalg.norm(obstacles[:, :2] - candidate[:2], axis=1)
            )
            candidates.append((min_clearance >= self._obstacle_clearance, path_length, candidate))

        valid = [candidate for candidate in candidates if candidate[0]]
        selected = min(valid or candidates, key=lambda item: item[1])[2]
        return [selected, approach]

    def _move_staging_point(
        self,
        point: NDArray[np.float64],
        normal: NDArray[np.float64],
        obstacles: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Shift an outer staging point sideways when an obstacle occupies it."""
        if not len(obstacles):
            return point
        required_clearance = self._obstacle_clearance + self._avoidance_margin
        if np.min(np.linalg.norm(obstacles[:, :2] - point[:2], axis=1)) >= required_clearance:
            return point

        lateral = np.array([-normal[1], normal[0]], dtype=np.float64)
        lateral /= max(np.linalg.norm(lateral), 1e-9)
        candidates = []
        for sign in (1.0, -1.0):
            candidate = point.copy()
            candidate[:2] += sign * required_clearance * lateral
            clearance = np.min(np.linalg.norm(obstacles[:, :2] - candidate[:2], axis=1))
            candidates.append((clearance, candidate))
        return max(candidates, key=lambda item: item[0])[1]

    @staticmethod
    def _remove_duplicate_waypoints(
        waypoints: list[NDArray[np.float64]],
    ) -> NDArray[np.float64]:
        """Remove consecutive points that would give the interpolator duplicate times."""
        result = [np.asarray(waypoints[0], dtype=np.float64)]
        for waypoint in waypoints[1:]:
            waypoint = np.asarray(waypoint, dtype=np.float64)
            if np.linalg.norm(waypoint - result[-1]) > 1e-5:
                result.append(waypoint)
        return np.asarray(result)

    def _allocate_waypoint_times(self, waypoints: NDArray[np.float64]) -> NDArray[np.float64]:
        distances = np.linalg.norm(np.diff(waypoints, axis=0), axis=1)
        durations = np.maximum(distances / self._cruise_speed, self._minimum_segment_time)
        return np.concatenate(([0.0], np.cumsum(durations)))

    @staticmethod
    def _quaternion_distance(a: NDArray[np.float64], b: NDArray[np.float64]) -> float:
        """Return the unsigned angular distance between two unit quaternions."""
        dot = float(np.clip(abs(a @ b), 0.0, 1.0))
        return 2.0 * np.arccos(dot)

    def _world_model_changed(self, obs: dict[str, NDArray[np.floating]]) -> bool:
        """Update newly visible objects and report whether the active local plan is stale."""
        old_gate_visited = self._gate_visited.copy()
        old_obstacles_visited = self._obstacles_visited.copy()
        self._gate_visited = np.asarray(obs["gates_visited"], dtype=bool).copy()
        self._obstacles_visited = np.asarray(obs["obstacles_visited"], dtype=bool).copy()
        self._remember_observed_objects(obs)

        active_gate_newly_seen = (
            self._current_gate_id >= 0
            and self._gate_visited[self._current_gate_id]
            and not old_gate_visited[self._current_gate_id]
        )
        obstacle_newly_seen = np.any(self._obstacles_visited & ~old_obstacles_visited)

        if self._current_gate_id in self._known_gates:
            center, quaternion = self._known_gates[self._current_gate_id]
            gate_pose_changed = (
                np.linalg.norm(center - self._planned_gate_pos) > 0.01
                or self._quaternion_distance(quaternion, self._planned_gate_quat) > 0.02
            )
        else:
            gate_pose_changed = False
        return bool(active_gate_newly_seen or obstacle_newly_seen or gate_pose_changed)

    def _activate_pending_gate_if_ready(
        self, obs: dict[str, NDArray[np.floating]]
    ) -> bool:
        """Finish the current exit segment before switching to the next gate."""
        if self._pending_pass_index is None:
            return False
        exit_reached = np.linalg.norm(np.asarray(obs["pos"]) - self._current_exit) < (
            self._exit_reached_distance
        )
        local_plan_finished = self._trajectory_time >= self._trajectory_duration
        if not (exit_reached or local_plan_finished):
            return False

        self._active_pass_index = self._pending_pass_index
        self._pending_pass_index = None
        self._plan_current_gate(obs)
        return True

    def compute_control(
        self, obs: dict[str, NDArray[np.floating]], info: dict | None = None
    ) -> NDArray[np.float32]:
        """Update the local plan and return a 16-dimensional full-state command."""
        del info

        switched_gate = self._activate_pending_gate_if_ready(obs)
        if (
            not switched_gate
            and self._pending_pass_index is None
            and self._world_model_changed(obs)
        ):
            self._plan_current_gate(obs)

        tracking_error = np.linalg.norm(
            np.asarray(obs["pos"], dtype=np.float64) - self._last_desired_position
        )
        time_scale = np.clip(
            1.0 - tracking_error / self._tracking_slowdown_distance, 0.15, 1.0
        )
        self._trajectory_time = min(
            self._trajectory_time + time_scale / self._freq, self._trajectory_duration
        )

        assert self._local_spline is not None
        desired_position = self._local_spline(self._trajectory_time)
        desired_velocity = self._local_spline(self._trajectory_time, 1) * time_scale
        desired_acceleration = self._local_spline(self._trajectory_time, 2) * time_scale**2
        self._last_desired_position = desired_position.copy()

        horizontal_velocity = desired_velocity[:2]
        if np.linalg.norm(horizontal_velocity) > 1e-4:
            self._desired_yaw = float(np.arctan2(horizontal_velocity[1], horizontal_velocity[0]))
        desired_quaternion = R.from_euler("z", self._desired_yaw).as_quat()

        return np.concatenate(
            (
                desired_position,
                desired_velocity,
                desired_acceleration,
                desired_quaternion,
                np.zeros(3),
            ),
            dtype=np.float32,
        )

    def step_callback(
        self,
        action: NDArray[np.floating],
        obs: dict[str, NDArray[np.floating]],
        reward: float,
        terminated: bool,
        truncated: bool,
        info: dict,
    ) -> bool:
        """Record gate progress while allowing the current gate exit segment to finish."""
        del action, reward, terminated, truncated, info
        gates_passed = int(obs["n_gates_passed"])
        total_passes = len(obs["gate_sequence"])
        self._finished = gates_passed >= total_passes
        if not self._finished and gates_passed > self._active_pass_index:
            self._pending_pass_index = gates_passed
        return self._finished

    def episode_callback(self):
        """No cross-episode learning is used by this baseline."""

    def render_callback(self, sim: Sim):
        """Draw the active local plan and its current setpoint."""
        assert self._local_spline is not None
        setpoint = self._local_spline(self._trajectory_time).reshape(1, -1)
        trajectory = self._local_spline(
            np.linspace(0.0, self._trajectory_duration, 100)
        )
        draw_line(sim, trajectory, rgba=(0.0, 1.0, 0.0, 1.0))
        draw_points(sim, setpoint, rgba=(1.0, 0.0, 0.0, 1.0), size=0.025)
