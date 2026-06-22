#!/usr/bin/env python3

import math
import signal

import rospy
from geometry_msgs.msg import PoseStamped, Twist
from std_srvs.srv import Trigger, TriggerResponse

from controllers import LevelRateController, ZVelocityController, clamp


class CopilotState:
    def __init__(self):
        self.x = 0.0
        self.y = 0.0
        self.z = 0.0
        self.vx = 0.0
        self.vy = 0.0
        self.vz = 0.0
        self.roll = 0.0
        self.pitch = 0.0
        self.yaw = 0.0
        self.last_pos_time = None
        self.has_pose = False
        self.has_velocity = False
        self.has_attitude = False


class CrazyflieCopilot:
    MODE_IDLE = "idle"
    MODE_HOLD = "hold"
    MODE_CMD = "cmd"
    MODE_TAKEOFF = "takeoff"
    MODE_LAND = "land"

    def __init__(self, cf_id):
        self.cf_id = cf_id
        self.ns = f"/cf{cf_id}"
        self.state = CopilotState()
        self.mode = self.MODE_IDLE
        self.last_cmd = Twist()
        self.last_cmd_time = rospy.Time.now()
        self.last_control_time = rospy.Time.now()
        self.takeoff_start_time = None
        self.takeoff_start_z = 0.0
        self.takeoff_settle_start_time = None
        self.xy_hold_x = 0.0
        self.xy_hold_y = 0.0
        self.has_xy_hold = False

        self.use_body_rate = rospy.get_param("~use_body_rate", True)
        self.control_rate = rospy.get_param("~copilot_control_rate", 50.0)
        self.cmd_timeout = rospy.get_param("~copilot_cmd_timeout", 0.3)
        self.max_z_velocity_up = abs(rospy.get_param("~max_z_velocity_up", 0.5))
        self.max_z_velocity_down = abs(rospy.get_param("~max_z_velocity_down", 0.4))

        self.takeoff_height = rospy.get_param("~takeoff_height", 0.5)
        self.takeoff_vz = rospy.get_param("~takeoff_vz", 0.35)
        self.takeoff_z_kp = rospy.get_param("~takeoff_z_kp", 1.0)
        self.takeoff_settle_error = rospy.get_param("~takeoff_settle_error", 0.03)
        self.takeoff_settle_time = rospy.get_param("~takeoff_settle_time", 1.0)
        self.xy_hold_kp = rospy.get_param("~xy_hold_kp", 1.5)
        self.xy_hold_kd = rospy.get_param("~xy_hold_kd", 1.0)
        self.xy_hold_angle_limit = rospy.get_param("~xy_hold_angle_limit", 0.25)
        self.land_vz = rospy.get_param("~land_vz", -0.25)
        self.land_stop_height = rospy.get_param("~land_stop_height", 0.08)

        min_thrust = rospy.get_param("~min_thrust", 10000)
        max_thrust = rospy.get_param("~max_thrust", 60000)
        self.adapt_max_accel_on_takeoff = rospy.get_param("~adapt_max_accel_on_takeoff", True)
        self.max_accel_adapt_kp = rospy.get_param("~max_accel_adapt_kp", 2.0)
        self.z_controller = ZVelocityController(
            max_thrust=max_thrust,
            max_accel=rospy.get_param("~max_accel", 18.86),
            kp=rospy.get_param("~z_vel_kp", 4.0),
            min_thrust=min_thrust,
            gravity=rospy.get_param("~gravity", 9.81),
            min_accel=rospy.get_param("~min_accel", 13.0),
            max_accel_limit=rospy.get_param("~max_accel_limit", 21.0),
        )

        self.level_rate_controller = LevelRateController(
            roll_kp=rospy.get_param("~level_roll_rate_kp", 2.0),
            pitch_kp=rospy.get_param("~level_pitch_rate_kp", 2.0),
            rate_limit=rospy.get_param("~level_rate_limit", 2.0),
        )

        self.cmd_pub = rospy.Publisher(f"{self.ns}/_cmd_vel", Twist, queue_size=1)
        rospy.Subscriber(f"{self.ns}/cmd_vel", Twist, self._cmd_cb, queue_size=1, tcp_nodelay=True)
        mocap_topic = self._resolve_mocap_pose_topic(cf_id)
        rospy.Subscriber(mocap_topic, PoseStamped, self._mocap_cb, queue_size=1, tcp_nodelay=True)

        rospy.Service(f"{self.ns}/takeoff", Trigger, self._takeoff_srv)
        rospy.Service(f"{self.ns}/land", Trigger, self._land_srv)

        period = 1.0 / self.control_rate
        self.timer = rospy.Timer(rospy.Duration(period), self._control_cb)
        rospy.loginfo(f"[cf{self.cf_id}] z velocity mode enabled, mocap topic: {mocap_topic}")

    def _cmd_cb(self, msg):
        self.last_cmd = msg
        self.last_cmd_time = rospy.Time.now()
        if self.mode not in (self.MODE_TAKEOFF, self.MODE_LAND):
            self.mode = self.MODE_CMD

    def _resolve_mocap_pose_topic(self, cf_id):
        topic = rospy.get_param("~mocap_pose_topic", None)
        if topic is None:
            topic = rospy.get_param("~vrpn_pose_topic", f"/vrpn_client_node/cf{cf_id}/pose")
        return topic.format(id=cf_id, cf_id=cf_id)

    def _mocap_cb(self, msg):
        now = msg.header.stamp if msg.header.stamp != rospy.Time(0) else rospy.Time.now()
        p = msg.pose.position
        if self.state.last_pos_time is not None:
            dt = (now - self.state.last_pos_time).to_sec()
            if dt > 0.0:
                self.state.vx = (p.x - self.state.x) / dt
                self.state.vy = (p.y - self.state.y) / dt
                self.state.vz = (p.z - self.state.z) / dt
                self.state.has_velocity = True

        self.state.x = p.x
        self.state.y = p.y
        self.state.z = p.z
        self.state.last_pos_time = now
        self.state.has_pose = True

        q = msg.pose.orientation
        self.state.roll, self.state.pitch, self.state.yaw = self._quat_to_euler(
            q.x, q.y, q.z, q.w
        )
        self.state.has_attitude = True

    def _quat_to_euler(self, x, y, z, w):
        sinr_cosp = 2.0 * (w * x + y * z)
        cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
        roll = math.atan2(sinr_cosp, cosr_cosp)

        sinp = 2.0 * (w * y - z * x)
        if abs(sinp) >= 1.0:
            pitch = math.copysign(math.pi / 2.0, sinp)
        else:
            pitch = math.asin(sinp)

        siny_cosp = 2.0 * (w * z + x * y)
        cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
        yaw = math.atan2(siny_cosp, cosy_cosp)
        return roll, pitch, yaw

    def _takeoff_srv(self, _):
        if not self.state.has_pose:
            return TriggerResponse(False, "copilot has no position estimate yet")
        self.mode = self.MODE_TAKEOFF
        self.takeoff_start_time = rospy.Time.now()
        self.takeoff_start_z = self.state.z
        self.takeoff_settle_start_time = None
        self.xy_hold_x = self.state.x
        self.xy_hold_y = self.state.y
        self.has_xy_hold = True
        self.z_controller.reset()
        self.level_rate_controller.reset()
        return TriggerResponse(
            True,
            f"takeoff to ({self.xy_hold_x:.2f}, {self.xy_hold_y:.2f}, {self.takeoff_height:.2f})",
        )

    def _land_srv(self, _):
        self.mode = self.MODE_LAND
        self.takeoff_start_time = None
        self.takeoff_settle_start_time = None
        self.xy_hold_x = self.state.x
        self.xy_hold_y = self.state.y
        self.has_xy_hold = self.state.has_pose
        self.z_controller.reset()
        self.level_rate_controller.reset()
        return TriggerResponse(True, "landing")

    def _control_cb(self, _):
        now = rospy.Time.now()
        dt = (now - self.last_control_time).to_sec()
        self.last_control_time = now

        if not self.state.has_pose:
            self._publish_stop()
            return

        cmd_age = (now - self.last_cmd_time).to_sec()
        if self.mode == self.MODE_CMD and cmd_age > self.cmd_timeout:
            self.mode = self.MODE_IDLE
            self.z_controller.reset()
            self.level_rate_controller.reset()

        if self.mode == self.MODE_TAKEOFF:
            roll_cmd, pitch_cmd = self._xy_hold_commands(dt)
            yawrate_ref = 0.0
            z_ref = self._takeoff_z_ref(now)
            ramp_done = z_ref >= self.takeoff_height
            if ramp_done:
                vz_ref = self._takeoff_hold_vz_ref(z_ref)
            else:
                vz_ref = self.takeoff_vz

            if self.adapt_max_accel_on_takeoff:
                self.z_controller.adapt_max_accel(
                    z_ref, self.state.z, dt, self.max_accel_adapt_kp
                )

            if self._takeoff_is_settled(now, ramp_done):
                self.mode = self.MODE_HOLD
                self.takeoff_start_time = None
                self.takeoff_settle_start_time = None
                vz_ref = 0.0
                rospy.loginfo(
                    f"[cf{self.cf_id}] takeoff complete, holding position, max_accel={self.z_controller.max_accel:.2f}"
                )
        elif self.mode == self.MODE_LAND:
            roll_cmd, pitch_cmd = self._xy_hold_commands(dt)
            vz_ref = self.land_vz
            yawrate_ref = 0.0
            if self.state.z <= self.land_stop_height:
                self.mode = self.MODE_IDLE
                self._publish_stop()
                return
        elif self.mode == self.MODE_HOLD:
            roll_cmd, pitch_cmd = self._xy_hold_commands(dt)
            vz_ref = self._takeoff_hold_vz_ref(self.takeoff_height)
            yawrate_ref = 0.0
        elif self.mode == self.MODE_CMD:
            roll_cmd = self.last_cmd.angular.x
            pitch_cmd = self.last_cmd.angular.y
            vz_ref = self.last_cmd.linear.z
            yawrate_ref = self.last_cmd.angular.z
        else:
            self._publish_stop()
            return

        vz_ref = self._limit_z_velocity(vz_ref)
        roll, pitch = self._tilt_for_thrust(roll_cmd, pitch_cmd)
        thrust = self.z_controller.update(vz_ref, self.state.vz, roll, pitch, dt)

        out = Twist()
        out.angular.x = roll_cmd
        out.angular.y = pitch_cmd
        out.angular.z = yawrate_ref
        out.linear.z = thrust
        self.cmd_pub.publish(out)

    def _level_commands(self, dt):
        if not self.use_body_rate:
            return 0.0, 0.0
        if not self.state.has_attitude:
            return 0.0, 0.0
        return self.level_rate_controller.update(
            0.0, 0.0, self.state.roll, self.state.pitch, dt
        )

    def _xy_hold_commands(self, dt):
        if not self.has_xy_hold:
            return self._level_commands(dt)

        ax_ref = self.xy_hold_kp * (self.xy_hold_x - self.state.x) - self.xy_hold_kd * self.state.vx
        ay_ref = self.xy_hold_kp * (self.xy_hold_y - self.state.y) - self.xy_hold_kd * self.state.vy

        vertical_accel = max(1.0, self.z_controller.gravity)
        rd_x = ax_ref / vertical_accel
        rd_y = ay_ref / vertical_accel

        yaw = self.state.yaw if self.state.has_attitude else 0.0
        cos_yaw = math.cos(yaw)
        sin_yaw = math.sin(yaw)
        pitch_ref = cos_yaw * rd_x + sin_yaw * rd_y
        roll_ref = sin_yaw * rd_x - cos_yaw * rd_y

        roll_ref = clamp(roll_ref, -self.xy_hold_angle_limit, self.xy_hold_angle_limit)
        pitch_ref = clamp(pitch_ref, -self.xy_hold_angle_limit, self.xy_hold_angle_limit)

        if not self.use_body_rate:
            return roll_ref, pitch_ref
        if not self.state.has_attitude:
            return 0.0, 0.0
        return self.level_rate_controller.update(
            roll_ref, pitch_ref, self.state.roll, self.state.pitch, dt
        )

    def _takeoff_z_ref(self, now):
        if self.takeoff_start_time is None:
            return self.takeoff_height
        elapsed = (now - self.takeoff_start_time).to_sec()
        return min(self.takeoff_height, self.takeoff_start_z + self.takeoff_vz * elapsed)

    def _takeoff_hold_vz_ref(self, z_ref):
        z_error = z_ref - self.state.z
        return clamp(
            self.takeoff_z_kp * z_error,
            -abs(self.takeoff_vz),
            abs(self.takeoff_vz),
        )

    def _limit_z_velocity(self, vz_ref):
        return clamp(vz_ref, -self.max_z_velocity_down, self.max_z_velocity_up)

    def _takeoff_is_settled(self, now, ramp_done):
        if not ramp_done:
            self.takeoff_settle_start_time = None
            return False

        z_error = self.takeoff_height - self.state.z
        if abs(z_error) > self.takeoff_settle_error:
            self.takeoff_settle_start_time = None
            return False

        if self.takeoff_settle_start_time is None:
            self.takeoff_settle_start_time = now
            return False

        settled_for = (now - self.takeoff_settle_start_time).to_sec()
        return settled_for >= self.takeoff_settle_time

    def _tilt_for_thrust(self, roll_cmd, pitch_cmd):
        if self.state.has_attitude:
            return self.state.roll, self.state.pitch
        if not self.use_body_rate:
            return roll_cmd, pitch_cmd
        return 0.0, 0.0

    def _publish_stop(self):
        self.cmd_pub.publish(Twist())


def parse_ids(raw):
    if isinstance(raw, str):
        raw = raw.strip("[]")
        return [int(x) for x in raw.replace(",", " ").split()]
    return list(raw)


def main():
    rospy.init_node("crazyflie_copilot")
    if not rospy.get_param("~use_z_velocity", False):
        rospy.loginfo("z velocity mode disabled; copilot not started")
        return

    ids = parse_ids(rospy.get_param("~ids", [1]))
    if not ids:
        rospy.logfatal("~ids is empty")
        return

    copilots = [CrazyflieCopilot(cf_id) for cf_id in ids]
    signal.signal(signal.SIGINT, lambda *_: rospy.signal_shutdown("SIGINT"))
    rospy.spin()
    for copilot in copilots:
        copilot._publish_stop()


if __name__ == "__main__":
    main()
