#!/usr/bin/env python3

import time
import sys
import signal
import math

import rospy
from geometry_msgs.msg import Twist, Vector3Stamped, PoseStamped
from std_msgs.msg import Bool, Float32
from std_srvs.srv import SetBool, SetBoolResponse, Trigger, TriggerResponse

import cflib
from cflib.crazyflie import Crazyflie
from cflib.crazyflie.log import LogConfig
from cflib.utils import uri_helper

def id_to_uri(cf_id: int) -> str:
    """Return a radio URI given a numeric Crazyflie ID (1…8)."""
    table = {
        1: "radio://0/10/2M/E7E7E7E701",
        2: "radio://0/10/2M/E7E7E7E702",
        3: "radio://0/10/2M/E7E7E7E703",
        4: "radio://0/10/2M/E7E7E7E704",
        5: "radio://0/20/2M/E7E7E7E705",
        6: "radio://0/20/2M/E7E7E7E706",
        7: "radio://0/20/2M/E7E7E7E707",
        8: "radio://0/50/2M/E7E7E7E708",
        9: "radio://0/80/2M/E7E7E7E709",
    }
    if cf_id not in table:
        raise ValueError(f"Invalid Crazyflie ID {cf_id}")
    return table[cf_id]

class CrazyflieServer:
    def __init__(self, cf_id: int):
        self.ns = f"/cf{cf_id}"
        self.cf_id = cf_id

        # ---- parameters ----
        self.use_body_rate          = rospy.get_param("~use_body_rate", True)
        self.pos_LOG                = rospy.get_param("~pos_LOG", False)
        self.vel_LOG                = rospy.get_param("~vel_LOG", False)
        self.ang_LOG                = rospy.get_param("~ang_LOG", False)
        self.thrust_LOG             = rospy.get_param("~thrust_LOG", False)
        self.acc_LOG                = rospy.get_param("~acc_LOG", False)
        self.gyro_raw_LOG           = rospy.get_param("~gyro_raw_LOG", False)
        self.gyro_LOG               = rospy.get_param("~gyro_LOG", False)
        self.stabilizer_controller  = rospy.get_param("~stabilizer_controller", 1) # 1 = PID, 2 = Mellinger, 3 = INDI, 4 = Brescianini
        self.stabilizer_estimator   = rospy.get_param("~stabilizer_estimator", 2)  # 1 = Complementary filter, 2 = EKF, 3 = unscented Kalman filter
        self.use_z_velocity         = rospy.get_param("~use_z_velocity", False)
        default_command_topic       = "_cmd_vel" if self.use_z_velocity else "cmd_vel"
        self.command_topic          = rospy.get_param("~command_topic", default_command_topic)
        self.auto_arm               = rospy.get_param("~auto_arm", self.use_z_velocity)
        self.armed                  = False
        self.killed                 = False
                
        # ---- Crazyflie link ----
        cflib.crtp.init_drivers(enable_debug_driver=False)
        self._cf = Crazyflie()

        # callbacks
        self._cf.connected        .add_callback(self._connected)
        self._cf.disconnected     .add_callback(self._disconnected)
        self._cf.connection_failed.add_callback(self._conn_failed)
        self._cf.connection_lost  .add_callback(self._conn_lost)

        uri = uri_helper.uri_from_env(default=id_to_uri(cf_id))
        self._cf.open_link(uri)

        # unlock startup thrust protection
        self._cf.commander.send_setpoint(0, 0, 0, 0)

        # ---- ROS pubs/subs ----
        def topic(name): return f"{self.ns}/{name}"

        rospy.Subscriber(topic(self.command_topic), Twist, self._twist_cb)
        rospy.loginfo(f"[cf{self.cf_id}] listening for commands on {topic(self.command_topic)}")
        self.arm_srv = rospy.Service(topic("arm"), SetBool, self._arm_srv)
        self.kill_srv = rospy.Service(topic("kill"), Trigger, self._kill_srv)
        self.reset_kill_srv = rospy.Service(topic("reset_kill"), Trigger, self._reset_kill_srv)

        if self.vel_LOG:             self.pub_vel          = rospy.Publisher(topic("crazyflieVel"),           Vector3Stamped, queue_size=10)
        if self.ang_LOG:             self.pub_ang          = rospy.Publisher(topic("crazyflieAng"),           Vector3Stamped, queue_size=10)
        if self.thrust_LOG:          self.pub_thrust       = rospy.Publisher(topic("crazyflieThrust"),        Vector3Stamped, queue_size=10)
        if self.acc_LOG:             self.pub_acc          = rospy.Publisher(topic("crazyflieAcc"),           Vector3Stamped, queue_size=10)
        if self.gyro_raw_LOG:        self.pub_gyro_raw     = rospy.Publisher(topic("crazyflieRawAngRate"),    Vector3Stamped, queue_size=10)
        if self.gyro_LOG:            self.pub_gyro         = rospy.Publisher(topic("crazyflieAngRate"),       Vector3Stamped, queue_size=10)
        if self.pos_LOG:             self.pub_pos          = rospy.Publisher(topic("crazyfliePos"),           Vector3Stamped, queue_size=10)

        self.pub_is_flying       = rospy.Publisher(topic("crazyflieIsFlying"),       Bool,           queue_size=10)
        self.pub_can_fly         = rospy.Publisher(topic("crazyflieCanFly"),         Bool,           queue_size=10)
        self.pub_z_range         = rospy.Publisher(topic("crazyflieZRange"),         Vector3Stamped, queue_size=10)
        self.pub_battery_voltage = rospy.Publisher(topic("crazyflieBatteryVoltage"), Float32,        queue_size=10)
        self.pub_battery_level   = rospy.Publisher(topic("crazyflieBatteryLevel"),   Float32,        queue_size=10)
        

        # flags to print “log started” once
        self._flags = {}

    # ───── connection callbacks ────────────────────────────────────────────────
    def _connected(self, uri):
        rospy.loginfo(f"[cf{self.cf_id}] connected on {uri}")
        rospy.Timer(rospy.Duration(2.0), self._init_cf, oneshot=True)

    def _init_cf(self, _):
        # set params
        try:
            self._cf.param.set_value("stabilizer.controller", self.stabilizer_controller)
            self._cf.param.set_value("stabilizer.estimator",  self.stabilizer_estimator)
            if self.use_body_rate:
                self._cf.param.set_value("flightmode.stabModeRoll", 0)
                self._cf.param.set_value("flightmode.stabModePitch",0)
                self._cf.param.set_value("flightmode.stabModeYaw",  0)
            else:
                self._cf.param.set_value("flightmode.stabModeRoll", 1)
                self._cf.param.set_value("flightmode.stabModePitch",1)
                self._cf.param.set_value("flightmode.stabModeYaw",  0)
        except Exception as e:
            rospy.logwarn(f"[cf{self.cf_id}] param set failed: {e}")

        if self.auto_arm:
            self._set_armed(True, "startup")
        else:
            rospy.loginfo(f"[cf{self.cf_id}] waiting for {self.ns}/arm before accepting commands")
        
        # configure logs
        self._setup_logs()

    def _conn_failed(self, uri, msg): rospy.logerr(f"[cf{self.cf_id}] connection failed: {msg}")
    def _conn_lost(self, uri, msg):   rospy.logerr(f"[cf{self.cf_id}] connection lost: {msg}")
    def _disconnected(self, uri):     rospy.loginfo(f"[cf{self.cf_id}] disconnected")

    # ---- logs ----
    def _setup_logs(self):
        # small helper
        def start_log(logconf, data_cb):
            try:
                self._cf.log.add_config(logconf)
                logconf.data_received_cb.add_callback(data_cb)
                logconf.error_cb.add_callback(
                    lambda lc, m: rospy.logerr(f"[cf{self.cf_id}] log {lc.name}: {m}")
                )
                logconf.start()
            except KeyError as e:
                rospy.logwarn(f"[cf{self.cf_id}] log {logconf.name}: {e}")

        # supervisor / battery / range are always on
        lg_sup = LogConfig("Supervisor", 1000)
        lg_sup.add_variable("supervisor.info", "uint16_t")
        start_log(lg_sup, self._cb_supervisor)

        lg_bat = LogConfig("Battery", 1000)
        lg_bat.add_variable("pm.vbat", "float")
        lg_bat.add_variable("pm.batteryLevel", "int8_t")
        start_log(lg_bat, self._cb_battery)

        lg_range = LogConfig("Range", 33)
        lg_range.add_variable("range.zrange", "float")
        start_log(lg_range, self._cb_range)

        # optional logs
        if self.pos_LOG:
            lg = LogConfig("Position", 16)
            for p in ("stateEstimate.x", "stateEstimate.y", "stateEstimate.z"):
                lg.add_variable(p, "float")
            start_log(lg, self._cb_position)

        if self.vel_LOG:
            lg = LogConfig("Velocity", 16)
            for v in ("stateEstimate.vx", "stateEstimate.vy", "stateEstimate.vz"):
                lg.add_variable(v, "float")
            start_log(lg, self._cb_velocity)

        if self.ang_LOG:
            lg = LogConfig("Angle", 16)
            for a in ("stateEstimate.roll", "stateEstimate.pitch", "stateEstimate.yaw"):
                lg.add_variable(a, "float")
            start_log(lg, self._cb_angle)

        if self.thrust_LOG:
            lg = LogConfig("Thrust", 16)
            lg.add_variable("stabilizer.thrust", "float")
            start_log(lg, self._cb_thrust)

        if self.acc_LOG:
            lg = LogConfig("Accel", 16)
            for a in ("acc.x", "acc.y", "acc.z"):
                lg.add_variable(a, "float")
            start_log(lg, self._cb_accel)

        if self.gyro_raw_LOG:
            lg = LogConfig("GyroRaw", 16)
            for g in ("gyro.xRaw", "gyro.yRaw", "gyro.zRaw"):
                lg.add_variable(g, "int16_t")
            start_log(lg, self._cb_gyro_raw)

        if self.gyro_LOG:
            lg = LogConfig("Gyro", 16)
            for g in ("gyro.x", "gyro.y", "gyro.z"):
                lg.add_variable(g, "float")
            start_log(lg, self._cb_gyro)

    # ---- log callbacks (publish to ROS) ----
    def _once(self, key, txt):
        if not self._flags.get(key, False):
            rospy.loginfo(f"[cf{self.cf_id}] {txt}")
            self._flags[key] = True

    def _cb_supervisor(self, ts, data, _):
        self._once("sup", "Supervisor log started")
        info = data["supervisor.info"]
        self.pub_can_fly.publish(bool((info >> 3) & 1))
        self.pub_is_flying.publish(bool((info >> 4) & 1))

    def _cb_battery(self, ts, data, _):
        self._once("bat", "Battery log started")
        self.pub_battery_voltage.publish(Float32(data["pm.vbat"]))
        self.pub_battery_level.publish(Float32(float(data["pm.batteryLevel"])))

    def _cb_range(self, ts, data, _):
        self._once("range", "Range log started")
        z = data["range.zrange"] / 1000.0  # mm→m
        msg = Vector3Stamped()
        msg.header.stamp = rospy.Time.now()
        msg.vector.z = z
        self.pub_z_range.publish(msg)

    def _cb_position(self, ts, data, _):
        self._once("pos", "Position log started")
        # estimated position
        pos = Vector3Stamped()
        pos.header.stamp = rospy.Time.now()
        pos.vector.x = data["stateEstimate.x"]
        pos.vector.y = data["stateEstimate.y"]
        pos.vector.z = data["stateEstimate.z"]
        self.pub_pos.publish(pos)

    def _cb_velocity(self, ts, data, _):
        self._once("vel", "Velocity log started")
        msg = Vector3Stamped()
        msg.header.stamp = rospy.Time.now()
        msg.vector.x = data["stateEstimate.vx"]
        msg.vector.y = data["stateEstimate.vy"]
        msg.vector.z = data["stateEstimate.vz"]
        self.pub_vel.publish(msg)

    def _cb_angle(self, ts, data, _):
        self._once("ang", "Angle log started")
        deg2rad = 0.01745329252
        msg = Vector3Stamped()
        msg.header.stamp = rospy.Time.now()
        msg.vector.x = data["stateEstimate.roll"]  * deg2rad
        msg.vector.y = -data["stateEstimate.pitch"]* deg2rad
        msg.vector.z = data["stateEstimate.yaw"]   * deg2rad
        self.pub_ang.publish(msg)

    def _cb_thrust(self, ts, data, _):
        self._once("thr", "Thrust log started")
        msg = Vector3Stamped()
        msg.header.stamp = rospy.Time.now()
        msg.vector.z = data["stabilizer.thrust"]
        self.pub_thrust.publish(msg)

    def _cb_accel(self, ts, data, _):
        self._once("acc", "Acceleration log started")
        g = 9.81
        msg = Vector3Stamped()
        msg.header.stamp = rospy.Time.now()
        msg.vector.x = -data["acc.x"] * g
        msg.vector.y = -data["acc.y"] * g
        msg.vector.z = -data["acc.z"] * g
        self.pub_acc.publish(msg)

    def _cb_gyro_raw(self, ts, data, _):
        self._once("gyro_raw", "Raw Gyro log started")
        scale = 0.001065
        msg = Vector3Stamped()
        msg.header.stamp = rospy.Time.now()
        msg.vector.x = data["gyro.xRaw"] * scale
        msg.vector.y = data["gyro.yRaw"] * scale
        msg.vector.z = data["gyro.zRaw"] * scale
        self.pub_gyro_raw.publish(msg)

    def _cb_gyro(self, ts, data, _):
        self._once("gyro", "Gyro log started")
        deg2rad = 0.01745329252
        msg = Vector3Stamped()
        msg.header.stamp = rospy.Time.now()
        msg.vector.x = data["gyro.x"] * deg2rad
        msg.vector.y = data["gyro.y"] * deg2rad
        msg.vector.z = data["gyro.z"] * deg2rad
        self.pub_gyro.publish(msg)

    # ---- cmd_vel subscriber ----
    def _twist_cb(self, msg: Twist):
        if self.killed:
            rospy.logwarn_throttle(1.0, f"[cf{self.cf_id}] ignoring command: kill is active")
            self._send_emergency_stop(repeats=1)
            return
        if not self.armed:
            rospy.logwarn_throttle(1.0, f"[cf{self.cf_id}] ignoring command: not armed")
            self._cf.commander.send_setpoint(0, 0, 0, 0)
            return

        if not math.isfinite(msg.linear.z):
            rospy.logwarn(f"[cf{self.cf_id}] received invalid thrust")
            self._cf.commander.send_setpoint(0, 0, 0, 0)
            return

        thrust = int(max(0, min(60000, msg.linear.z)))
        if self.use_body_rate:
            rollrate  =  msg.angular.x * 57.2958
            pitchrate =  msg.angular.y * 57.2958
            yawrate   = -msg.angular.z * 57.2958
            self._cf.commander.send_setpoint(rollrate, pitchrate, yawrate, thrust)
        else:
            roll   =  msg.angular.x * 57.2958
            pitch  =  msg.angular.y * 57.2958
            yawrate= -msg.angular.z * 57.2958
            self._cf.commander.send_setpoint(roll, pitch, yawrate, thrust)

    # ---- arm service ----
    def _arm_srv(self, req):
        ok, message = self._set_armed(req.data, "service")
        return SetBoolResponse(ok, message)

    def _set_armed(self, armed, reason):
        if armed and self.killed:
            return False, f"[cf{self.cf_id}] cannot arm while kill is active"

        try:
            if not armed:
                self._cf.commander.send_setpoint(0, 0, 0, 0)
                try:
                    self._cf.commander.send_stop_setpoint()
                except Exception:
                    pass

            self._cf.platform.send_arming_request(bool(armed))
            self.armed = bool(armed)
            state = "armed" if self.armed else "disarmed"
            rospy.logwarn(f"[cf{self.cf_id}] {state} by {reason}")
            return True, f"[cf{self.cf_id}] {state}"
        except Exception as e:
            return False, f"[cf{self.cf_id}] arm request failed: {e}"

    # ---- emergency services ----
    def _kill_srv(self, _):
        self.kill("service")
        return TriggerResponse(True, f"[cf{self.cf_id}] kill active")

    def _reset_kill_srv(self, _):
        self.killed = False
        self.armed = False
        try:
            self._cf.commander.send_setpoint(0, 0, 0, 0)
        except Exception as e:
            return TriggerResponse(False, f"[cf{self.cf_id}] reset_kill failed: {e}")
        rospy.logwarn(f"[cf{self.cf_id}] kill reset; call /cf{self.cf_id}/arm before commands")
        return TriggerResponse(True, f"[cf{self.cf_id}] kill reset")

    def kill(self, reason="kill"):
        self.killed = True
        self.armed = False
        rospy.logerr(f"[cf{self.cf_id}] KILL triggered by {reason}")
        self._send_emergency_stop(repeats=5)

    def _send_emergency_stop(self, repeats=3):
        for _ in range(repeats):
            try:
                self._cf.commander.send_setpoint(0, 0, 0, 0)
            except Exception:
                pass
            try:
                self._cf.commander.send_stop_setpoint()
            except Exception:
                pass
            try:
                self._cf.platform.send_arming_request(False)
            except Exception:
                pass
            self.armed = False
            time.sleep(0.02)

    # ---- cleanup ----
    def close(self):
        try:
            self._send_emergency_stop(repeats=2)
            self._cf.close_link()
        except Exception:
            pass

def main():
    rospy.init_node("multi_crazyflie_server")

    raw = rospy.get_param("~ids", [1])
    if isinstance(raw, str):
        raw = raw.strip("[]")
        ids = [int(x) for x in raw.replace(',', ' ').split()]
    else:
        ids = list(raw)
    if not ids:
        rospy.logfatal("~ids is empty")
        sys.exit(1)

    servers = [CrazyflieServer(cf_id=i) for i in ids]

    def _kill_all_srv(_):
        for s in servers:
            s.kill("kill_all")
        return TriggerResponse(True, "kill active for all Crazyflies")

    kill_all_srv = rospy.Service("kill_all", Trigger, _kill_all_srv)

    # ---------- graceful shutdown ------------------------------------------
    def _close_everything():
        rospy.loginfo("Shutting down links...")
        for s in servers:
            s.close()           # closes USB / radio, signals cflib threads
        time.sleep(0.2)          # give cflib worker threads time to exit

    rospy.on_shutdown(_close_everything)

    # Catch Ctrl-C ourselves and forward it to rospy
    signal.signal(signal.SIGINT, lambda *_: rospy.signal_shutdown("SIGINT"))

    # -----------------------------------------------------------------------
    rospy.spin()                 # returns when signal_shutdown() called
    _close_everything()          # second call is harmless
    sys.exit(0)

if __name__ == "__main__":
    main()
