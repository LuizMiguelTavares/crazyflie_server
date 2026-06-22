#!/usr/bin/env python3

import math


def clamp(value, low, high):
    return max(low, min(high, value))


class LinearController:
    def __init__(self, kp, ki=0.0, kd=0.0, output_limit=None, integrator_limit=None):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.output_limit = output_limit
        self.integrator_limit = integrator_limit
        self.integral = 0.0
        self.prev_error = None

    def reset(self):
        self.integral = 0.0
        self.prev_error = None

    def update(self, reference, measurement, dt):
        if dt <= 0.0 or not math.isfinite(dt):
            return 0.0

        error = reference - measurement
        self.integral += error * dt
        if self.integrator_limit is not None:
            self.integral = clamp(self.integral, -self.integrator_limit, self.integrator_limit)

        derivative = 0.0
        if self.prev_error is not None:
            derivative = (error - self.prev_error) / dt
        self.prev_error = error

        output = self.kp * error + self.ki * self.integral + self.kd * derivative
        if self.output_limit is not None:
            output = clamp(output, -self.output_limit, self.output_limit)
        return output


class ZVelocityController:
    def __init__(
        self,
        max_thrust,
        max_accel,
        kp,
        min_thrust,
        gravity=9.81,
        min_accel=13.0,
        max_accel_limit=21.0,
    ):
        self.max_thrust = max_thrust
        self.max_accel = max_accel
        self.kp = kp
        self.gravity = gravity
        self.min_thrust = min_thrust
        self.min_accel = min_accel
        self.max_accel_limit = max_accel_limit

    def reset(self):
        pass

    def adapt_max_accel(self, z_ref, z, dt, kp):
        if dt <= 0.0 or not math.isfinite(dt):
            return
        z_error = z_ref - z
        self.max_accel -= kp * z_error * dt
        self.max_accel = clamp(self.max_accel, self.min_accel, self.max_accel_limit)

    def update(self, vz_ref, vz, roll, pitch, dt):
        az_ref = self.kp * (vz_ref - vz)
        az_ref = clamp(az_ref, -self.gravity, self.max_accel - self.gravity)
        tilt_compensation = math.cos(roll) * math.cos(pitch)
        tilt_compensation = max(0.5, tilt_compensation)
        thrust_per_accel = self.max_thrust / self.max_accel
        thrust = (self.gravity + az_ref) * thrust_per_accel / tilt_compensation
        return int(clamp(thrust, self.min_thrust, self.max_thrust))


class LevelRateController:
    def __init__(self, roll_kp, pitch_kp, rate_limit):
        self.roll_controller = LinearController(roll_kp, output_limit=rate_limit)
        self.pitch_controller = LinearController(pitch_kp, output_limit=rate_limit)

    def reset(self):
        self.roll_controller.reset()
        self.pitch_controller.reset()

    def update(self, roll_ref, pitch_ref, roll, pitch, dt):
        roll_rate = self.roll_controller.update(roll_ref, roll, dt)
        pitch_rate = self.pitch_controller.update(pitch_ref, pitch, dt)
        return roll_rate, pitch_rate
