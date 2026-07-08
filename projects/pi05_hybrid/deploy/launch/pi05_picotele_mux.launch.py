"""Integrated launch for Pi0.5 VLA, picotele, RealSense, and command mux.

Run this file from an environment where the picotele and realsense workspaces
have already been sourced. The launch keeps hardware ownership inside
picotele while routing both teleop and VLA commands through /mux/*.
"""

from pathlib import Path

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PythonExpression
from launch_ros.actions import Node


def _pi05_script(project_dir: Path, script_name: str, config, condition):
    return ExecuteProcess(
        cmd=[
            str(project_dir / "deploy" / "scripts" / script_name),
            config,
        ],
        output="screen",
        condition=IfCondition(condition),
    )


def generate_launch_description():
    project_dir = Path(__file__).resolve().parents[2]
    default_config = str(project_dir / "deploy" / "config" / "deploy.yaml")
    default_realsense_launch = "/home/hit/octopus/octopus/launch/realsense_triple_compressed.launch.py"

    config = LaunchConfiguration("config")
    launch_realsense = LaunchConfiguration("launch_realsense")
    launch_pico = LaunchConfiguration("launch_pico")
    launch_picotele = LaunchConfiguration("launch_picotele")
    launch_vla = LaunchConfiguration("launch_vla")
    launch_bridge = LaunchConfiguration("launch_bridge")
    launch_mux = LaunchConfiguration("launch_mux")
    use_external_planner = LaunchConfiguration("use_external_planner")

    right_arm_ip = LaunchConfiguration("right_arm_ip")
    left_arm_ip = LaunchConfiguration("left_arm_ip")
    arm_port = LaunchConfiguration("arm_port")
    gripper_transport = LaunchConfiguration("gripper_transport")
    right_port = LaunchConfiguration("right_port")
    left_port = LaunchConfiguration("left_port")
    right_gripper_ip = LaunchConfiguration("right_gripper_ip")
    left_gripper_ip = LaunchConfiguration("left_gripper_ip")
    gripper_port = LaunchConfiguration("gripper_port")
    position_scale = LaunchConfiguration("position_scale")
    vertical_scale = LaunchConfiguration("vertical_scale")
    max_translation = LaunchConfiguration("max_translation")
    vertical_max_translation = LaunchConfiguration("vertical_max_translation")
    loop_hz = LaunchConfiguration("loop_hz")
    ik_hz = LaunchConfiguration("ik_hz")
    state_pub_hz = LaunchConfiguration("state_pub_hz")
    hand_state_pub_hz = LaunchConfiguration("hand_state_pub_hz")
    hand_command_hz = LaunchConfiguration("hand_command_hz")
    hand_tactile_hz = LaunchConfiguration("hand_tactile_hz")
    home_on_start = LaunchConfiguration("home_on_start")
    require_deadman = LaunchConfiguration("require_deadman")
    grip_threshold = LaunchConfiguration("grip_threshold")
    pose_timeout_sec = LaunchConfiguration("pose_timeout_sec")
    planner_target_timeout_s = LaunchConfiguration("planner_target_timeout_s")
    max_linear_step = LaunchConfiguration("max_linear_step")
    vertical_max_linear_step = LaunchConfiguration("vertical_max_linear_step")
    max_angular_step_deg = LaunchConfiguration("max_angular_step_deg")
    orientation_deadband_deg = LaunchConfiguration("orientation_deadband_deg")
    max_orientation_offset_deg = LaunchConfiguration("max_orientation_offset_deg")
    orientation_mode = LaunchConfiguration("orientation_mode")
    wrist_roll_translation_threshold = LaunchConfiguration("wrist_roll_translation_threshold")
    wrist_roll_axis = LaunchConfiguration("wrist_roll_axis")
    wrist_roll_scale = LaunchConfiguration("wrist_roll_scale")
    disable_grippers = LaunchConfiguration("disable_grippers")
    disable_orientation = LaunchConfiguration("disable_orientation")
    enable_state_udp_push = LaunchConfiguration("enable_state_udp_push")
    enable_state_poll_fallback = LaunchConfiguration("enable_state_poll_fallback")
    state_udp_cycle_multiple = LaunchConfiguration("state_udp_cycle_multiple")
    right_state_udp_port = LaunchConfiguration("right_state_udp_port")
    left_state_udp_port = LaunchConfiguration("left_state_udp_port")
    right_state_udp_target_ip = LaunchConfiguration("right_state_udp_target_ip")
    left_state_udp_target_ip = LaunchConfiguration("left_state_udp_target_ip")
    udp_state_timeout_s = LaunchConfiguration("udp_state_timeout_s")
    udp_state_min_jump_reject_rad = LaunchConfiguration("udp_state_min_jump_reject_rad")
    udp_state_jump_rate_factor = LaunchConfiguration("udp_state_jump_rate_factor")
    udp_state_jump_margin_rad = LaunchConfiguration("udp_state_jump_margin_rad")
    udp_state_repeat_warn_streak = LaunchConfiguration("udp_state_repeat_warn_streak")
    udp_state_stats_log_interval_s = LaunchConfiguration("udp_state_stats_log_interval_s")
    enable_controller_safety_config = LaunchConfiguration("enable_controller_safety_config")
    controller_collision_stage = LaunchConfiguration("controller_collision_stage")
    enable_soft_stop_hooks = LaunchConfiguration("enable_soft_stop_hooks")
    enable_singularity_check = LaunchConfiguration("enable_singularity_check")
    singularity_distance_threshold = LaunchConfiguration("singularity_distance_threshold")
    enable_active_singularity_avoidance = LaunchConfiguration("enable_active_singularity_avoidance")
    singularity_warn_distance_m = LaunchConfiguration("singularity_warn_distance_m")
    singularity_stop_distance_m = LaunchConfiguration("singularity_stop_distance_m")
    singularity_universal_value_limit = LaunchConfiguration("singularity_universal_value_limit")
    singularity_ik_seed_offset_deg = LaunchConfiguration("singularity_ik_seed_offset_deg")
    singularity_projection_steps = LaunchConfiguration("singularity_projection_steps")
    singularity_path_check_samples = LaunchConfiguration("singularity_path_check_samples")
    singularity_speed_min_scale = LaunchConfiguration("singularity_speed_min_scale")
    enable_collision_heuristic = LaunchConfiguration("enable_collision_heuristic")
    collision_current_threshold = LaunchConfiguration("collision_current_threshold")
    collision_joint_error_threshold = LaunchConfiguration("collision_joint_error_threshold")
    collision_confirm_cycles = LaunchConfiguration("collision_confirm_cycles")
    enable_watchdog = LaunchConfiguration("enable_watchdog")
    watchdog_hz = LaunchConfiguration("watchdog_hz")

    picotele_arguments = [
        "--right-arm-ip",
        right_arm_ip,
        "--left-arm-ip",
        left_arm_ip,
        "--arm-port",
        arm_port,
        "--gripper-transport",
        gripper_transport,
        "--right-port",
        right_port,
        "--left-port",
        left_port,
        "--right-gripper-ip",
        right_gripper_ip,
        "--left-gripper-ip",
        left_gripper_ip,
        "--gripper-port",
        gripper_port,
        "--position-scale",
        position_scale,
        "--vertical-scale",
        vertical_scale,
        "--max-translation",
        max_translation,
        "--vertical-max-translation",
        vertical_max_translation,
        "--loop-hz",
        loop_hz,
        "--ik-hz",
        ik_hz,
        "--use-external-planner",
        use_external_planner,
        "--planner-target-timeout-s",
        planner_target_timeout_s,
        "--state-pub-hz",
        state_pub_hz,
        "--hand-state-pub-hz",
        hand_state_pub_hz,
        "--hand-command-hz",
        hand_command_hz,
        "--hand-tactile-hz",
        hand_tactile_hz,
        "--home-on-start",
        home_on_start,
        "--require-deadman",
        require_deadman,
        "--grip-threshold",
        grip_threshold,
        "--pose-timeout-sec",
        pose_timeout_sec,
        "--max-linear-step",
        max_linear_step,
        "--vertical-max-linear-step",
        vertical_max_linear_step,
        "--max-angular-step-deg",
        max_angular_step_deg,
        "--orientation-deadband-deg",
        orientation_deadband_deg,
        "--max-orientation-offset-deg",
        max_orientation_offset_deg,
        "--orientation-mode",
        orientation_mode,
        "--wrist-roll-translation-threshold",
        wrist_roll_translation_threshold,
        "--wrist-roll-axis",
        wrist_roll_axis,
        "--wrist-roll-scale",
        wrist_roll_scale,
        "--disable-grippers",
        disable_grippers,
        "--disable-orientation",
        disable_orientation,
        "--enable-state-udp-push",
        enable_state_udp_push,
        "--enable-state-poll-fallback",
        enable_state_poll_fallback,
        "--state-udp-cycle-multiple",
        state_udp_cycle_multiple,
        "--right-state-udp-port",
        right_state_udp_port,
        "--left-state-udp-port",
        left_state_udp_port,
        "--right-state-udp-target-ip",
        right_state_udp_target_ip,
        "--left-state-udp-target-ip",
        left_state_udp_target_ip,
        "--udp-state-timeout-s",
        udp_state_timeout_s,
        "--udp-state-min-jump-reject-rad",
        udp_state_min_jump_reject_rad,
        "--udp-state-jump-rate-factor",
        udp_state_jump_rate_factor,
        "--udp-state-jump-margin-rad",
        udp_state_jump_margin_rad,
        "--udp-state-repeat-warn-streak",
        udp_state_repeat_warn_streak,
        "--udp-state-stats-log-interval-s",
        udp_state_stats_log_interval_s,
        "--enable-controller-safety-config",
        enable_controller_safety_config,
        "--controller-collision-stage",
        controller_collision_stage,
        "--enable-soft-stop-hooks",
        enable_soft_stop_hooks,
        "--enable-singularity-check",
        enable_singularity_check,
        "--singularity-distance-threshold",
        singularity_distance_threshold,
        "--enable-active-singularity-avoidance",
        enable_active_singularity_avoidance,
        "--singularity-warn-distance-m",
        singularity_warn_distance_m,
        "--singularity-stop-distance-m",
        singularity_stop_distance_m,
        "--singularity-universal-value-limit",
        singularity_universal_value_limit,
        "--singularity-ik-seed-offset-deg",
        singularity_ik_seed_offset_deg,
        "--singularity-projection-steps",
        singularity_projection_steps,
        "--singularity-path-check-samples",
        singularity_path_check_samples,
        "--singularity-speed-min-scale",
        singularity_speed_min_scale,
        "--enable-collision-heuristic",
        enable_collision_heuristic,
        "--collision-current-threshold",
        collision_current_threshold,
        "--collision-joint-error-threshold",
        collision_joint_error_threshold,
        "--collision-confirm-cycles",
        collision_confirm_cycles,
        "--enable-watchdog",
        enable_watchdog,
        "--watchdog-hz",
        watchdog_hz,
    ]

    picotele_condition = IfCondition(launch_picotele)
    planner_condition = IfCondition(
        PythonExpression(
            [
                "'",
                launch_picotele,
                "' == 'true' and '",
                use_external_planner,
                "' == 'true'",
            ]
        )
    )

    return LaunchDescription(
        [
            DeclareLaunchArgument("config", default_value=default_config),
            DeclareLaunchArgument("launch_realsense", default_value="false"),
            DeclareLaunchArgument("realsense_launch", default_value=default_realsense_launch),
            DeclareLaunchArgument("launch_pico", default_value="true"),
            DeclareLaunchArgument("launch_picotele", default_value="true"),
            DeclareLaunchArgument("launch_vla", default_value="true"),
            DeclareLaunchArgument("launch_bridge", default_value="true"),
            DeclareLaunchArgument("launch_mux", default_value="true"),
            DeclareLaunchArgument("right_arm_ip", default_value="192.168.1.18"),
            DeclareLaunchArgument("left_arm_ip", default_value="192.168.1.19"),
            DeclareLaunchArgument("arm_port", default_value="8080"),
            DeclareLaunchArgument("gripper_transport", default_value="serial"),
            DeclareLaunchArgument("right_port", default_value="/dev/ttyUSB1"),
            DeclareLaunchArgument("left_port", default_value="/dev/ttyUSB0"),
            DeclareLaunchArgument("right_gripper_ip", default_value="192.168.1.20"),
            DeclareLaunchArgument("left_gripper_ip", default_value="192.168.1.21"),
            DeclareLaunchArgument("gripper_port", default_value="6000"),
            DeclareLaunchArgument("position_scale", default_value="1.5"),
            DeclareLaunchArgument("vertical_scale", default_value="1.0"),
            DeclareLaunchArgument("max_translation", default_value="0.25"),
            DeclareLaunchArgument("vertical_max_translation", default_value="0.8"),
            DeclareLaunchArgument("loop_hz", default_value="60.0"),
            DeclareLaunchArgument("ik_hz", default_value="60.0"),
            DeclareLaunchArgument("use_external_planner", default_value="true"),
            DeclareLaunchArgument("planner_target_timeout_s", default_value="0.15"),
            DeclareLaunchArgument("state_pub_hz", default_value="100.0"),
            DeclareLaunchArgument("hand_state_pub_hz", default_value="60.0"),
            DeclareLaunchArgument("hand_command_hz", default_value="60.0"),
            DeclareLaunchArgument("hand_tactile_hz", default_value="30.0"),
            DeclareLaunchArgument("home_on_start", default_value="true"),
            DeclareLaunchArgument("require_deadman", default_value="true"),
            DeclareLaunchArgument("grip_threshold", default_value="0.3"),
            DeclareLaunchArgument("pose_timeout_sec", default_value="5.0"),
            DeclareLaunchArgument("max_linear_step", default_value="0.01"),
            DeclareLaunchArgument("vertical_max_linear_step", default_value="0.025"),
            DeclareLaunchArgument("max_angular_step_deg", default_value="5.0"),
            DeclareLaunchArgument("orientation_deadband_deg", default_value="5.0"),
            DeclareLaunchArgument("max_orientation_offset_deg", default_value="35.0"),
            DeclareLaunchArgument("orientation_mode", default_value="tool_roll"),
            DeclareLaunchArgument("wrist_roll_translation_threshold", default_value="0.01"),
            DeclareLaunchArgument("wrist_roll_axis", default_value="z"),
            DeclareLaunchArgument("wrist_roll_scale", default_value="-1.5"),
            DeclareLaunchArgument("disable_grippers", default_value="false"),
            DeclareLaunchArgument("disable_orientation", default_value="false"),
            DeclareLaunchArgument("enable_state_udp_push", default_value="true"),
            DeclareLaunchArgument("enable_state_poll_fallback", default_value="true"),
            DeclareLaunchArgument("state_udp_cycle_multiple", default_value="2"),
            DeclareLaunchArgument("right_state_udp_port", default_value="8089"),
            DeclareLaunchArgument("left_state_udp_port", default_value="8090"),
            DeclareLaunchArgument("right_state_udp_target_ip", default_value=""),
            DeclareLaunchArgument("left_state_udp_target_ip", default_value=""),
            DeclareLaunchArgument("udp_state_timeout_s", default_value="1.0"),
            DeclareLaunchArgument("udp_state_min_jump_reject_rad", default_value="0.25"),
            DeclareLaunchArgument("udp_state_jump_rate_factor", default_value="5.0"),
            DeclareLaunchArgument("udp_state_jump_margin_rad", default_value="0.05"),
            DeclareLaunchArgument("udp_state_repeat_warn_streak", default_value="20"),
            DeclareLaunchArgument("udp_state_stats_log_interval_s", default_value="5.0"),
            DeclareLaunchArgument("enable_controller_safety_config", default_value="false"),
            DeclareLaunchArgument("controller_collision_stage", default_value="4"),
            DeclareLaunchArgument("enable_soft_stop_hooks", default_value="false"),
            DeclareLaunchArgument("enable_singularity_check", default_value="false"),
            DeclareLaunchArgument("singularity_distance_threshold", default_value="0.04"),
            DeclareLaunchArgument("enable_active_singularity_avoidance", default_value="true"),
            DeclareLaunchArgument("singularity_warn_distance_m", default_value="0.08"),
            DeclareLaunchArgument("singularity_stop_distance_m", default_value="0.04"),
            DeclareLaunchArgument("singularity_universal_value_limit", default_value="0.015"),
            DeclareLaunchArgument("singularity_ik_seed_offset_deg", default_value="15.0"),
            DeclareLaunchArgument("singularity_projection_steps", default_value="12"),
            DeclareLaunchArgument("singularity_path_check_samples", default_value="8"),
            DeclareLaunchArgument("singularity_speed_min_scale", default_value="0.20"),
            DeclareLaunchArgument("enable_collision_heuristic", default_value="false"),
            DeclareLaunchArgument("collision_current_threshold", default_value="5.0"),
            DeclareLaunchArgument("collision_joint_error_threshold", default_value="0.30"),
            DeclareLaunchArgument("collision_confirm_cycles", default_value="3"),
            DeclareLaunchArgument("enable_watchdog", default_value="false"),
            DeclareLaunchArgument("watchdog_hz", default_value="1.0"),
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource([LaunchConfiguration("realsense_launch")]),
                condition=IfCondition(launch_realsense),
            ),
            Node(
                package="pico",
                executable="pico",
                name="pico",
                output="screen",
                condition=IfCondition(launch_pico),
            ),
            Node(
                package="picotele",
                executable="picotele_planner_node",
                name="picotele_planner_node",
                output="screen",
                arguments=picotele_arguments,
                remappings=[
                    (
                        "/picotele/left_arm/safe_joint_target",
                        "/teleop/left_arm/safe_joint_target",
                    ),
                    (
                        "/picotele/right_arm/safe_joint_target",
                        "/teleop/right_arm/safe_joint_target",
                    ),
                ],
                condition=planner_condition,
            ),
            Node(
                package="picotele",
                executable="pico_teleop_node",
                name="picotele_arm_node",
                output="screen",
                arguments=picotele_arguments,
                remappings=[
                    (
                        "/picotele/left_arm/safe_joint_target",
                        "/mux/left_arm/safe_joint_target",
                    ),
                    (
                        "/picotele/right_arm/safe_joint_target",
                        "/mux/right_arm/safe_joint_target",
                    ),
                    ("/xr/pico/left/grip", "/mux/left_arm/deadman"),
                    ("/xr/pico/right/grip", "/mux/right_arm/deadman"),
                ],
                condition=picotele_condition,
            ),
            Node(
                package="picotele",
                executable="pico_hand_node",
                name="picotele_hand_node",
                output="screen",
                arguments=picotele_arguments,
                remappings=[
                    ("/xr/pico/left/trigger", "/mux/left_hand/trigger"),
                    ("/xr/pico/right/trigger", "/mux/right_hand/trigger"),
                ],
                condition=picotele_condition,
            ),
            Node(
                package="picotele",
                executable="pico_tactile_node",
                name="picotele_tactile_node",
                output="screen",
                arguments=picotele_arguments,
                condition=picotele_condition,
            ),
            _pi05_script(project_dir, "run_command_mux.sh", config, launch_mux),
            _pi05_script(project_dir, "run_bridge.sh", config, launch_bridge),
            _pi05_script(project_dir, "run_inference.sh", config, launch_vla),
        ]
    )
