import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.io import loadmat
from matplotlib import patches, transforms
from animator_math import physical_wheelbase, road_xy, unscale_vehicle_states
import argparse


class DataProcessor:
    """Handles all data loading and processing"""
    
    def __init__(self,leader_file, follower_file, track_file):
        self.Leader = leader_file
        self.Follower = follower_file
        self.Track = track_file
        self.load_data(leader_file, follower_file)
        self.process_data()
        
    def load_data(self,Leader,Follower):
        """Load and extract raw data from MAT files"""
        leader = loadmat(Leader, squeeze_me=True, struct_as_record=False)
        follower = loadmat(Follower, squeeze_me=True, struct_as_record=False)

        self.auxdata = follower["output"].result.setup.auxdata
        self.lengthscale = self.auxdata.lengthscale
        self.massscale = self.auxdata.massscale
        self.timescale = self.auxdata.timescale
        self.velscale = self.lengthscale / self.timescale
        self.forcescale = self.massscale * self.lengthscale / self.timescale**2
        self.accelscale = self.lengthscale / self.timescale**2

        # Vehicle parameters
        self.a = self.auxdata.a
        self.b = self.auxdata.b
        self.a_m, self.b_m = physical_wheelbase(self.auxdata)
        self.M = self.auxdata.M  # Vehicle mass
        self.rwTrack = self.auxdata.track.rw / self.lengthscale
        self.xc = self.auxdata.track.xc
        self.yc = self.auxdata.track.yc
        self.psiTrack = self.auxdata.track.psi
        self.sTrack = self.auxdata.track.s / self.lengthscale

        # Tyre model parameters
        self._extract_tyre_parameters()

        # Normal loads
        controlF = follower["output"].result.interpsolution.phase.control
        controlL = leader["output"].result.interpsolution.phase.control
        self.Ffz = controlF[:, 3] 
        self.Frz = controlF[:, 4] 
        self.FfzL = controlL[:, 3] 
        self.FrzL = controlL[:, 4] 

        # Simulation results
        self.sL = leader["output"].result.interpsolution.phase.time / self.lengthscale
        self.sF = follower["output"].result.interpsolution.phase.time / self.lengthscale
        self.statesL = leader["output"].result.interpsolution.phase.state
        self.statesF = follower["output"].result.interpsolution.phase.state

        # Check if udot/vdot are available
        leader_phase = leader["output"].result.interpsolution.phase
        follower_phase = follower["output"].result.interpsolution.phase

        # Get field names 
        leader_fields = leader_phase._fieldnames if hasattr(leader_phase, '_fieldnames') else []
        follower_fields = follower_phase._fieldnames if hasattr(follower_phase, '_fieldnames') else []

        self.has_direct_accelerations = 'udot' in leader_fields and 'udot' in follower_fields

        if self.has_direct_accelerations:
            # Original data - use direct udot/vdot
            print("Loading original data with direct accelerations")
            self.udotL_raw = leader_phase.udot / self.accelscale
            self.udotF_raw = follower_phase.udot / self.accelscale
            self.vdotL_raw = leader_phase.vdot / self.accelscale
            self.vdotF_raw = follower_phase.vdot / self.accelscale
        else:
            # Real-world data - calculate from forces
            print("Loading real-world data - accelerations will be calculated from forces")
            self.udotL_raw = None
            self.udotF_raw = None
            self.vdotL_raw = None
            self.vdotF_raw = None
    
    def _compute_accelerations(self):
        """Compute vehicle accelerations"""
        def compute_derivative(y, t):
            dy = np.zeros_like(y)
            dy[0] = (y[1] - y[0]) / (t[1] - t[0])
            dy[1:-1] = (y[2:] - y[:-2]) / (t[2:] - t[:-2])
            dy[-1] = (y[-1] - y[-2]) / (t[-1] - t[-2])
            return dy

        # Leader accelerations numerical
        u_dot_L = compute_derivative(self.uL, self.tL)
        v_dot_L = compute_derivative(self.vL, self.tL)
        self.ax_L = u_dot_L - self.vL * self.omega_BzL
        self.ay_L = v_dot_L + self.uL * self.omega_BzL

        # Follower accelerations numerical
        u_dot_F = compute_derivative(self.uF, self.tF)
        v_dot_F = compute_derivative(self.vF, self.tF)
        self.ax_F = u_dot_F - self.vF * self.omega_BzF
        self.ay_F = v_dot_F + self.uF * self.omega_BzF

        # Directly from raw data (only if available)
        if self.has_direct_accelerations:
            self.ax_L_direct = self.udotL_raw - self.vL * self.omega_BzL
            self.ay_L_direct = self.vdotL_raw + self.uL * self.omega_BzL
            self.ax_F_direct = self.udotF_raw - self.vF * self.omega_BzF
            self.ay_F_direct = self.vdotF_raw + self.uF * self.omega_BzF
        else:
            # For real-world data, set direct accelerations to numerical values
            self.ax_L_direct = self.ax_L
            self.ay_L_direct = self.ay_L
            self.ax_F_direct = self.ax_F
            self.ay_F_direct = self.ay_F

    def _compute_accelerations_from_forces(self):
        """Compute accelerations from tyre forces for real-world data"""
        print("Computing accelerations from tyre forces...")
        
        # Total longitudinal and lateral forces
        Fx_total_F = self.Ffx_F_t + self.Frx_F_t
        Fy_total_F = self.Ffy_F_t + self.Fry_F_t
        Fx_total_L = self.Ffx_L_t + self.Frx_L_t
        Fy_total_L = self.Ffy_L_t + self.Fry_L_t
        
        # Convert to accelerations (F = m*a => a = F/m)
        self.ax_from_forces_F = Fx_total_F / self.M
        self.ay_from_forces_F = Fy_total_F / self.M
        self.ax_from_forces_L = Fx_total_L / self.M
        self.ay_from_forces_L = Fy_total_L / self.M
        
        # Convert to g-forces
        self.g_long_F = self.ax_from_forces_F / 9.81
        self.g_lat_F = self.ay_from_forces_F / 9.81
        self.g_long_L = self.ax_from_forces_L / 9.81
        self.g_lat_L = self.ay_from_forces_L / 9.81
        
        print("Force-based acceleration calculation complete")

    def get_acceleration_data(self, method='auto'):
        """
        Get acceleration data with specified method
        method: 'auto', 'direct', 'numerical', 'forces'
        """
        if method == 'auto':
            if self.has_direct_accelerations:
                return self.ax_L_direct, self.ay_L_direct, self.ax_F_direct, self.ay_F_direct
            else:
                return self.ax_L, self.ay_L, self.ax_F, self.ay_F
        elif method == 'direct':
            return self.ax_L_direct, self.ay_L_direct, self.ax_F_direct, self.ay_F_direct
        elif method == 'numerical':
            return self.ax_L, self.ay_L, self.ax_F, self.ay_F
        elif method == 'forces':
            if hasattr(self, 'ax_from_forces_L'):
                return self.ax_from_forces_L, self.ay_from_forces_L, self.ax_from_forces_F, self.ay_from_forces_F
            else:
                print("Force-based accelerations not available, using numerical")
                return self.ax_L, self.ay_L, self.ax_F, self.ay_F

    def get_g_forces(self, method='auto'):
        """Get g-forces with specified method"""
        ax_L, ay_L, ax_F, ay_F = self.get_acceleration_data(method)
        
        g_long_L = ax_L / 9.81
        g_lat_L = ay_L / 9.81
        g_long_F = ax_F / 9.81
        g_lat_F = ay_F / 9.81
        
        return g_long_L, g_lat_L, g_long_F, g_lat_F

    def _extract_tyre_parameters(self):
        """Extract tyre model parameters from auxdata"""
        # Front tyre parameters
        self.Fz1_F = self.auxdata.Fz1_F
        self.Fz2_F = self.auxdata.Fz2_F
        self.muxmax1_F = self.auxdata.muxmax1_F
        self.muxmax2_F = self.auxdata.muxmax2_F
        self.kmax1_F = self.auxdata.kmax1_F
        self.kmax2_F = self.auxdata.kmax2_F
        self.muymax1_F = self.auxdata.muymax1_F
        self.muymax2_F = self.auxdata.muymax2_F
        self.alpmax1_F = self.auxdata.alpmax1_F
        self.alpmax2_F = self.auxdata.alpmax2_F
        self.Qx_F = self.auxdata.Qx_F
        self.Qy_F = self.auxdata.Qy_F
        self.Sx_F = self.auxdata.Sx_F
        self.Sy_F = self.auxdata.Sy_F

        # Rear tyre parameters
        self.Fz1_R = self.auxdata.Fz1_R
        self.Fz2_R = self.auxdata.Fz2_R
        self.muxmax1_R = self.auxdata.muxmax1_R
        self.muxmax2_R = self.auxdata.muxmax2_R
        self.kmax1_R = self.auxdata.kmax1_R
        self.kmax2_R = self.auxdata.kmax2_R
        self.muymax1_R = self.auxdata.muymax1_R
        self.muymax2_R = self.auxdata.muymax2_R
        self.alpmax1_R = self.auxdata.alpmax1_R
        self.alpmax2_R = self.auxdata.alpmax2_R
        self.Qx_R = self.auxdata.Qx_R
        self.Qy_R = self.auxdata.Qy_R
        self.Sx_R = self.auxdata.Sx_R
        self.Sy_R = self.auxdata.Sy_R

    def process_data(self):
        """Process all the loaded data"""
        self._extract_states()
        self._compute_accelerations()
        self._compute_slip_angles()
        self._compute_tyre_forces()
        self._compute_positions()
        self._resample_to_common_time()
        self._compute_derived_quantities()

    def _extract_states(self):
        """Extract and scale vehicle states"""
        leader = unscale_vehicle_states(self.statesL, self.lengthscale, self.velscale, self.timescale)
        follower = unscale_vehicle_states(self.statesF, self.lengthscale, self.velscale, self.timescale)

        self.nL = leader["n"]
        self.xiL = leader["xi"]
        self.vL = leader["v"]
        self.omega_BzL = leader["omega_Bz"]
        self.uL = leader["u"]
        self.deltaL = leader["delta"]
        self.k_fL = leader["k_f"]
        self.k_rL = leader["k_r"]
        self.tL = leader["t"]

        self.nF = follower["n"]
        self.xiF = follower["xi"]
        self.vF = follower["v"]
        self.omega_BzF = follower["omega_Bz"]
        self.uF = follower["u"]
        self.deltaF = follower["delta"]
        self.k_fF = follower["k_f"]
        self.k_rF = follower["k_r"]
        self.tF = follower["t"]

    def _compute_slip_angles(self):
        """Compute tyre slip angles"""
        self.alp_fF = np.arctan2(self.vF + self.omega_BzF * self.a_m, self.uF) - self.deltaF
        self.alp_rF = np.arctan2(self.vF - self.omega_BzF * self.b_m, self.uF)
        self.alp_fL = np.arctan2(self.vL + self.omega_BzL * self.a_m, self.uL) - self.deltaL
        self.alp_rL = np.arctan2(self.vL - self.omega_BzL * self.b_m, self.uL)

    def _compute_tyre_forces(self):
        """Compute tyre forces using magic formula"""
        # Front axle calculations
        self._compute_front_axle_forces()
        # Rear axle calculations
        self._compute_rear_axle_forces()

    def _compute_front_axle_forces(self):
        """Compute front axle tyre forces"""
        muxmaxfF = (self.Ffz - self.Fz1_F) * (self.muxmax2_F - self.muxmax1_F) / (self.Fz2_F - self.Fz1_F) + self.muxmax1_F
        muymaxfF = (self.Ffz - self.Fz1_F) * (self.muymax2_F - self.muymax1_F) / (self.Fz2_F - self.Fz1_F) + self.muymax1_F
        kmaxfF = (self.Ffz - self.Fz1_F) * (self.kmax2_F - self.kmax1_F) / (self.Fz2_F - self.Fz1_F) + self.kmax1_F
        alpmaxfF = (self.Ffz - self.Fz1_F) * (self.alpmax2_F - self.alpmax1_F) / (self.Fz2_F - self.Fz1_F) + self.alpmax1_F

        muxmaxfL = (self.FfzL - self.Fz1_F) * (self.muxmax2_F - self.muxmax1_F) / (self.Fz2_F - self.Fz1_F) + self.muxmax1_F
        muymaxfL = (self.FfzL - self.Fz1_F) * (self.muymax2_F - self.muymax1_F) / (self.Fz2_F - self.Fz1_F) + self.muymax1_F
        kmaxfL = (self.FfzL - self.Fz1_F) * (self.kmax2_F - self.kmax1_F) / (self.Fz2_F - self.Fz1_F) + self.kmax1_F
        alpmaxfL = (self.FfzL - self.Fz1_F) * (self.alpmax2_F - self.alpmax1_F) / (self.Fz2_F - self.Fz1_F) + self.alpmax1_F

        knfF = np.divide(self.k_fF, kmaxfF, out=np.zeros_like(self.k_fF), where=kmaxfF != 0)
        alpnfF = np.divide(self.alp_fF, alpmaxfF, out=np.zeros_like(self.alp_fF), where=alpmaxfF != 0)
        rhofF = np.hypot(knfF, alpnfF)

        knfL = np.divide(self.k_fL, kmaxfL, out=np.zeros_like(self.k_fL), where=kmaxfL != 0)
        alpnfL = np.divide(self.alp_fL, alpmaxfL, out=np.zeros_like(self.alp_fL), where=alpmaxfL != 0)
        rhofL = np.hypot(knfL, alpnfL)

        muxfF = muxmaxfF * np.sin(self.Qx_F * np.arctan(self.Sx_F * rhofF))
        muyfF = muymaxfF * np.sin(self.Qy_F * np.arctan(self.Sy_F * rhofF))
        muxfL = muxmaxfL * np.sin(self.Qx_F * np.arctan(self.Sx_F * rhofL))
        muyfL = muymaxfL * np.sin(self.Qy_F * np.arctan(self.Sy_F * rhofL))

        rhofF_safe = np.maximum(rhofF, np.finfo(float).eps)
        rhofL_safe = np.maximum(rhofL, np.finfo(float).eps)
        self.Ffx_F_t = muxfF * self.Ffz * (knfF / rhofF_safe)
        self.Ffy_F_t = muyfF * self.Ffz * (alpnfF / rhofF_safe)
        self.Ffx_L_t = muxfL * self.FfzL * (knfL / rhofL_safe)
        self.Ffy_L_t = muyfL * self.FfzL * (alpnfL / rhofL_safe)

        self.Fxmax_f_F_t = muxmaxfF * self.Ffz
        self.Fymax_f_F_t = muymaxfF * self.Ffz
        self.Fxmax_f_L_t = muxmaxfL * self.FfzL
        self.Fymax_f_L_t = muymaxfL * self.FfzL

    def _compute_rear_axle_forces(self):
        """Compute rear axle tyre forces"""
        muxmaxrF = (self.Frz - self.Fz1_R) * (self.muxmax2_R - self.muxmax1_R) / (self.Fz2_R - self.Fz1_R) + self.muxmax1_R
        muymaxrF = (self.Frz - self.Fz1_R) * (self.muymax2_R - self.muymax1_R) / (self.Fz2_R - self.Fz1_R) + self.muymax1_R
        kmaxrF = (self.Frz - self.Fz1_R) * (self.kmax2_R - self.kmax1_R) / (self.Fz2_R - self.Fz1_R) + self.kmax1_R
        alpmaxrF = (self.Frz - self.Fz1_R) * (self.alpmax2_R - self.alpmax1_R) / (self.Fz2_R - self.Fz1_R) + self.alpmax1_R

        muxmaxrL = (self.FrzL - self.Fz1_R) * (self.muxmax2_R - self.muxmax1_R) / (self.Fz2_R - self.Fz1_R) + self.muxmax1_R
        muymaxrL = (self.FrzL - self.Fz1_R) * (self.muymax2_R - self.muymax1_R) / (self.Fz2_R - self.Fz1_R) + self.muymax1_R
        kmaxrL = (self.FrzL - self.Fz1_R) * (self.kmax2_R - self.kmax1_R) / (self.Fz2_R - self.Fz1_R) + self.kmax1_R
        alpmaxrL = (self.FrzL - self.Fz1_R) * (self.alpmax2_R - self.alpmax1_R) / (self.Fz2_R - self.Fz1_R) + self.alpmax1_R

        knrF = np.divide(self.k_rF, kmaxrF, out=np.zeros_like(self.k_rF), where=kmaxrF != 0)
        alpnrF = np.divide(self.alp_rF, alpmaxrF, out=np.zeros_like(self.alp_rF), where=alpmaxrF != 0)
        rhorF = np.hypot(knrF, alpnrF)

        knrL = np.divide(self.k_rL, kmaxrL, out=np.zeros_like(self.k_rL), where=kmaxrL != 0)
        alpnrL = np.divide(self.alp_rL, alpmaxrL, out=np.zeros_like(self.alp_rL), where=alpmaxrL != 0)
        rhorL = np.hypot(knrL, alpnrL)

        muxrF = muxmaxrF * np.sin(self.Qx_R * np.arctan(self.Sx_R * rhorF))
        muyrF = muymaxrF * np.sin(self.Qy_R * np.arctan(self.Sy_R * rhorF))
        muxrL = muxmaxrL * np.sin(self.Qx_R * np.arctan(self.Sx_R * rhorL))
        muyrL = muymaxrL * np.sin(self.Qy_R * np.arctan(self.Sy_R * rhorL))

        rhorF_safe = np.maximum(rhorF, np.finfo(float).eps)
        rhorL_safe = np.maximum(rhorL, np.finfo(float).eps)
        self.Frx_F_t = muxrF * self.Frz * (knrF / rhorF_safe)
        self.Fry_F_t = muyrF * self.Frz * (alpnrF / rhorF_safe)
        self.Frx_L_t = muxrL * self.FrzL * (knrL / rhorL_safe)
        self.Fry_L_t = muyrL * self.FrzL * (alpnrL / rhorL_safe)

        self.Fxmax_r_F_t = muxmaxrF * self.Frz
        self.Fymax_r_F_t = muymaxrF * self.Frz 
        self.Fxmax_r_L_t = muxmaxrL * self.FrzL
        self.Fymax_r_L_t = muymaxrL * self.FrzL

        
    
    def _compute_positions(self):
        """Compute vehicle global positions"""
        # Track boundaries
        self.x_inner = self.xc - (-self.rwTrack / 2) * np.sin(self.psiTrack)
        self.y_inner = self.yc + (-self.rwTrack / 2) * np.cos(self.psiTrack)
        self.x_outer = self.xc - (self.rwTrack / 2) * np.sin(self.psiTrack)
        self.y_outer = self.yc + (+self.rwTrack / 2) * np.cos(self.psiTrack)

        # Interpolated positions
        xcF = np.interp(self.sF, self.sTrack, self.xc)
        ycF = np.interp(self.sF, self.sTrack, self.yc)
        psiF = np.interp(self.sF, self.sTrack, self.psiTrack)

        self.xF, self.yF = road_xy(xcF, ycF, psiF, self.nF)

        xcL = np.interp(self.sL, self.sTrack, self.xc)
        ycL = np.interp(self.sL, self.sTrack, self.yc)
        psiL = np.interp(self.sL, self.sTrack, self.psiTrack)

        self.xL, self.yL = road_xy(xcL, ycL, psiL, self.nL)

        self.carAngleF = np.unwrap(psiF + self.xiF)
        self.carAngleL = np.unwrap(psiL + self.xiL)

    def _resample_to_common_time(self):
        """Resample all data to common time base"""
        tNum = len(self.tF)
        self.t = np.linspace(0, min(self.tF[-1], self.tL[-1]), tNum)
        self.sF_i = np.interp(self.t, self.tF, self.sF)
        self.nF_i = np.interp(self.t, self.tF, self.nF)
        self.sL_i = np.interp(self.t, self.tL, self.sL)
        self.nL_i = np.interp(self.t, self.tL, self.nL)

        # Interpolate tyre quantities to common time base
        self.Ffx_F = np.interp(self.t, self.tF, self.Ffx_F_t);  self.Ffy_F = np.interp(self.t, self.tF, self.Ffy_F_t)
        self.Ffx_L = np.interp(self.t, self.tL, self.Ffx_L_t);  self.Ffy_L = np.interp(self.t, self.tL, self.Ffy_L_t)
        self.Frx_F = np.interp(self.t, self.tF, self.Frx_F_t);  self.Fry_F = np.interp(self.t, self.tF, self.Fry_F_t)
        self.Frx_L = np.interp(self.t, self.tL, self.Frx_L_t);  self.Fry_L = np.interp(self.t, self.tL, self.Fry_L_t)

        self.Fxmax_f_F = np.interp(self.t, self.tF, self.Fxmax_f_F_t);  self.Fymax_f_F = np.interp(self.t, self.tF, self.Fymax_f_F_t)
        self.Fxmax_f_L = np.interp(self.t, self.tL, self.Fxmax_f_L_t);  self.Fymax_f_L = np.interp(self.t, self.tL, self.Fymax_f_L_t)
        self.Fxmax_r_F = np.interp(self.t, self.tF, self.Fxmax_r_F_t);  self.Fymax_r_F = np.interp(self.t, self.tF, self.Fymax_r_F_t)
        self.Fxmax_r_L = np.interp(self.t, self.tL, self.Fxmax_r_L_t);  self.Fymax_r_L = np.interp(self.t, self.tL, self.Fymax_r_L_t)

        # Interpolated positions. Reconstruct from common-timeline track
        # coordinates so the HUD and 3D renderer share the same geometry.
        xcF = np.interp(self.sF_i, self.sTrack, self.xc)
        ycF = np.interp(self.sF_i, self.sTrack, self.yc)
        psiF = np.interp(self.sF_i, self.sTrack, self.psiTrack)
        self.xF_interp, self.yF_interp = road_xy(xcF, ycF, psiF, self.nF_i)
        self.carAngleF_interp = np.interp(self.t, self.tF, self.carAngleF)

        xcL = np.interp(self.sL_i, self.sTrack, self.xc)
        ycL = np.interp(self.sL_i, self.sTrack, self.yc)
        psiL = np.interp(self.sL_i, self.sTrack, self.psiTrack)
        self.xL_interp, self.yL_interp = road_xy(xcL, ycL, psiL, self.nL_i)
        self.carAngleL_interp = np.interp(self.t, self.tL, self.carAngleL)

        # Interpolate accelerations to common time base
        self.ax_L_interp = np.interp(self.t, self.tL, self.ax_L)
        self.ay_L_interp = np.interp(self.t, self.tL, self.ay_L)
        self.ax_F_interp = np.interp(self.t, self.tF, self.ax_F)
        self.ay_F_interp = np.interp(self.t, self.tF, self.ay_F)
        self.ax_L_direct_interp = np.interp(self.t, self.tL, self.ax_L_direct)
        self.ay_L_direct_interp = np.interp(self.t, self.tL, self.ay_L_direct)
        self.ax_F_direct_interp = np.interp(self.t, self.tF, self.ax_F_direct)
        self.ay_F_direct_interp = np.interp(self.t, self.tF, self.ay_F_direct)

    def _compute_derived_quantities(self):
        """Compute derived quantities for telemetry"""
        # Interpolate states needed for telemetry
        self.uF_i = np.interp(self.t, self.tF, self.uF)
        self.vF_i = np.interp(self.t, self.tF, self.vF)
        self.wF_i = np.interp(self.t, self.tF, self.omega_BzF)
        self.uL_i = np.interp(self.t, self.tL, self.uL)
        self.vL_i = np.interp(self.t, self.tL, self.vL)
        self.wL_i = np.interp(self.t, self.tL, self.omega_BzL)
        self.sF_i = np.interp(self.t, self.tF, self.sF)
        self.sL_i = np.interp(self.t, self.tL, self.sL)
        self.dF_i = np.interp(self.t, self.tF, self.deltaF)
        self.dL_i = np.interp(self.t, self.tL, self.deltaL)

        # Steering parameters
        self.deltaF_i = self.dF_i
        self.deltaL_i = self.dL_i

        try:
            t_one_sec = self.t[0] + 1.0
            early = (self.t <= t_one_sec) & (np.hypot(self.uF_i, self.vF_i) < 0.5)
            self.STEER_ZERO_OFFSET_DEG = -np.degrees(np.mean(self.deltaF_i[early])) if np.any(early) else 0.0
        except Exception:
            self.STEER_ZERO_OFFSET_DEG = 0.0

        max_delta_deg = float(np.max(np.abs(np.degrees(self.deltaF_i)))) + 1e-6
        self.STEER_WHEEL_DEG_PER_RAD = 180.0 / np.radians(max_delta_deg)
        self.STEER_WHEEL_DEG_PER_RAD = np.clip(self.STEER_WHEEL_DEG_PER_RAD, 120.0, 1200.0)

        # Speeds and gaps
        self.spdF = np.hypot(self.uF_i, self.vF_i)
        self.spdL = np.hypot(self.uL_i, self.vL_i)
        self.gap_s = self.sF_i - self.sL_i
        self.gap_xy = np.hypot(self.xL_interp - self.xF_interp, self.yL_interp - self.yF_interp)

        # Bar display ranges
        self.AX_MAX = max(np.max(np.abs(self.ax_F_interp)), np.max(np.abs(self.ax_L_interp))) + 1e-6
        self.AY_MAX = max(np.max(np.abs(self.ay_F_interp)), np.max(np.abs(self.ay_L_interp))) + 1e-6
        self.AX_LIM = max(5.0, min(25.0, 1.2 * self.AX_MAX))
        self.AY_LIM = max(5.0, min(50.0, 1.2 * self.AY_MAX))

        # Steering wheel parameters
        self.spoke_len = 0.82
        self.base_spokes = np.deg2rad([90, 210, 330])
        self.STEER_VISUAL_SIGN = -1


class TelemetryDashboard:
    """Main telemetry dashboard that coordinates the animation"""

    def __init__(self, leader_file, follower_file, track_file):
        self.data_processor = DataProcessor(leader_file, follower_file, track_file)
        self.precomputed_data = PrecomputedData(self.data_processor)
        self.fig = None
        self.animation = None
        
        # HUD components
        self.mini_carF = None
        self.mini_carL = None
        self.spokes_lines = []
        self.center_mark = None
        self.steer_text = None
        self.gg_leader_dot = None
        self.gg_follower_dot = None
        self.dash_vals = {}
        self.bar_throttle = None
        self.bar_ax = None
        self.bar_ay = None
        self.front_dot_F = None
        self.front_dot_L = None
        self.rear_dot_F = None
        self.rear_dot_L = None
        
    def setup_dashboard(self):
        """Setup the complete dashboard"""
        print("Pre-computing ALL animation data...")
        self.precomputed_data.precompute_all()
        
        print("Setting up dashboard...")
        self._setup_figure()
        self._setup_minimap()
        self._setup_steering_wheel()
        self._setup_gg_diagram()
        self._setup_dashboard()
        self._setup_telemetry_bars()
        self._setup_friction_circles()
        
        # Connect scroll event
        self.fig.canvas.mpl_connect("scroll_event", self._on_scroll)
        
    def _setup_figure(self):
        """Setup the main figure"""
        self.fig = plt.figure(figsize=(16, 10), facecolor='#1a1a1a')
        
        
    def _setup_minimap(self):
        """Setup minimap component"""
        ax_minimap = self.fig.add_axes([0.005, 0.50, 0.15, 0.4])
        ax_minimap.set_title('Track Overview', fontsize=11, color='white', fontweight='bold', pad=8)
        ax_minimap.set_facecolor('#0a0a0a')
        ax_minimap.set_aspect('equal')
        ax_minimap.grid(True, alpha=0.15, color='#333333')
        ax_minimap.set_xticks([])
        ax_minimap.set_yticks([])
        for spine in ax_minimap.spines.values():
            spine.set_edgecolor('#555555')
            spine.set_linewidth(2)

        # Full track on minimap
        ax_minimap.plot(self.data_processor.xc, self.data_processor.yc, color='#666666', linestyle=':', linewidth=1.2, alpha=0.5)
        ax_minimap.plot(self.data_processor.x_inner, self.data_processor.y_inner, color='#00ff00', linewidth=1.5, alpha=0.6)
        ax_minimap.plot(self.data_processor.x_outer, self.data_processor.y_outer, color='#00ff00', linewidth=1.5, alpha=0.6)

        xmin = min(np.min(self.data_processor.x_inner), np.min(self.data_processor.x_outer)) - 10
        xmax = max(np.max(self.data_processor.x_inner), np.max(self.data_processor.x_outer)) + 10
        ymin = min(np.min(self.data_processor.y_inner), np.min(self.data_processor.y_outer)) - 10
        ymax = max(np.max(self.data_processor.y_inner), np.max(self.data_processor.y_outer)) + 10
        ax_minimap.set_xlim(xmin, xmax)
        ax_minimap.set_ylim(ymin, ymax)
        ax_minimap.invert_yaxis()

        # Car dots on minimap
        self.mini_carF, = ax_minimap.plot([], [], 'o', color='#ff3333', markersize=12, markeredgecolor='white', markeredgewidth=2)
        self.mini_carL, = ax_minimap.plot([], [], 'o', color='#3366ff', markersize=12, markeredgecolor='white', markeredgewidth=2)
        
    def _setup_steering_wheel(self):
        """Setup steering wheel component"""
        ax_wheel = self.fig.add_axes([0.785, 0.75, 0.17, 0.21])
        ax_wheel.set_title('Steering (Follower)', fontsize=11, color='#ff3333', fontweight='bold', pad=8)
        ax_wheel.set_facecolor('#0a0a0a')
        ax_wheel.set_aspect('equal')
        ax_wheel.set_xlim(-1.15, 1.15)
        ax_wheel.set_ylim(-1.15, 1.15)
        ax_wheel.set_xticks([])
        ax_wheel.set_yticks([])
        for spine in ax_wheel.spines.values():
            spine.set_edgecolor('#555555')
            spine.set_linewidth(2)

        # Professional 3D-style steering wheel
        wheel_outer = patches.Circle((0, 0), 1.0, fill=False, edgecolor='#cccccc', linewidth=4)
        wheel_inner = patches.Circle((0, 0), 0.88, fill=False, edgecolor='#999999', linewidth=2)
        wheel_hub = patches.Circle((0, 0), 0.18, color='#333333', edgecolor='#666666', linewidth=2)
        ax_wheel.add_patch(wheel_outer)
        ax_wheel.add_patch(wheel_inner)
        ax_wheel.add_patch(wheel_hub)

        # Three spokes
        base_spokes = np.deg2rad([90, 210, 330])
        spoke_len = 0.82
        for ang in base_spokes:
            x = np.array([0.0, spoke_len*np.cos(ang)])
            y = np.array([0.0, spoke_len*np.sin(ang)])
            ln, = ax_wheel.plot(x, y, color='#aaaaaa', linewidth=5, solid_capstyle='round')
            self.spokes_lines.append(ln)

        # Center indicator mark
        self.center_mark = ax_wheel.plot([0, 0], [0.22, 0.35], color='#ff0000', linewidth=4, solid_capstyle='round')[0]

        # Steering angle text
        self.steer_text = ax_wheel.text(0, -0.65, '', fontsize=13, color='#ffffff', ha='center', 
                                fontweight='bold', family='monospace',
                                bbox=dict(boxstyle='round,pad=0.5', facecolor='#2a2a2a', edgecolor='#555555', linewidth=2))
        
    def _setup_gg_diagram(self):
        """Setup G-G diagram component"""
        ax_gg = self.fig.add_axes([0.05, 0.12, 0.21, 0.26])
        ax_gg.set_facecolor('#0a0a0a')
        ax_gg.set_xlim(-self.data_processor.AY_LIM*1.1, self.data_processor.AY_LIM*1.1)
        ax_gg.set_ylim(-self.data_processor.AX_LIM*1.1, self.data_processor.AX_LIM*1.1)
        ax_gg.set_xlabel('← Lateral → (m/s²)', fontsize=10, color='white', fontweight='bold')
        ax_gg.set_ylabel('← Longitudinal (m/s²) →  ', fontsize=10, color='white', fontweight='bold')
        ax_gg.set_title('G-G Diagram', fontsize=11, color='white', fontweight='bold', pad=8)
        ax_gg.grid(True, alpha=0.2, color='#333333')
        ax_gg.axhline(0, color='#666666', linewidth=1.5)
        ax_gg.axvline(0, color='#666666', linewidth=1.5)
        ax_gg.tick_params(colors='white', labelsize=9)
        for spine in ax_gg.spines.values():
            spine.set_edgecolor('#555555')
            spine.set_linewidth(2)

        # Plot full trajectories
        ax_gg.plot(self.data_processor.ay_L, self.data_processor.ax_L, color='#3366ff', alpha=0.15, linewidth=1.2)
        ax_gg.plot(self.data_processor.ay_F, self.data_processor.ax_F, color='#ff3333', alpha=0.15, linewidth=1.2)

        # Current position dots
        self.gg_leader_dot, = ax_gg.plot([], [], 'o', color='#3366ff', markersize=10, markeredgecolor='white', markeredgewidth=2)
        self.gg_follower_dot, = ax_gg.plot([], [], 'o', color='#ff3333', markersize=10, markeredgecolor='white', markeredgewidth=2)
        ax_gg.legend([self.gg_leader_dot, self.gg_follower_dot], ['Leader', 'Follower'], 
                     loc='upper right', fontsize=9, facecolor='#2a2a2a', edgecolor='#555555', labelcolor='white')
        
    def _setup_dashboard(self):
        """Setup main dashboard display"""
        ax_dash = self.fig.add_axes([0.785, 0.34, 0.21, 0.37])
        ax_dash.set_facecolor('#0a0a0a')
        ax_dash.set_xticks([])
        ax_dash.set_yticks([])
        for spine in ax_dash.spines.values():
            spine.set_edgecolor('#555555')
            spine.set_linewidth(2)

        # Dashboard text elements
        self.dash_vals = {
            "time": ax_dash.text(0.50, 0.85, "", transform=ax_dash.transAxes, fontsize=18, 
                                 color='#00ff00', ha='center', fontweight='bold', family='monospace'),
            "spdF": ax_dash.text(0.50, 0.72, "", transform=ax_dash.transAxes, fontsize=14, 
                                 color='#ff3333', ha='center', fontweight='bold', family='monospace'),
            "spdL": ax_dash.text(0.50, 0.60, "", transform=ax_dash.transAxes, fontsize=14, 
                                 color='#3366ff', ha='center', fontweight='bold', family='monospace'),
            "gapS": ax_dash.text(0.50, 0.44, "", transform=ax_dash.transAxes, fontsize=13, 
                                 color='#ffff00', ha='center', fontweight='bold', family='monospace'),
            "gapD": ax_dash.text(0.50, 0.32, "", transform=ax_dash.transAxes, fontsize=11, 
                                 color='#aaaaaa', ha='center', fontweight='normal', family='monospace'),
            "p1": ax_dash.text(0.50, 0.12, "", transform=ax_dash.transAxes, fontsize=13, 
                               color='#ffffff', ha='center', fontweight='bold'),
        }

        # Labels
        ax_dash.text(0.27, 0.86, "TIME", transform=ax_dash.transAxes, fontsize=9, 
                     color='#888888', ha='center', fontweight='bold')
        ax_dash.text(0.05, 0.72, "F:", transform=ax_dash.transAxes, fontsize=12, 
                     color='#ff3333', ha='left', fontweight='bold')
        ax_dash.text(0.05, 0.60, "L:", transform=ax_dash.transAxes, fontsize=12, 
                     color='#3366ff', ha='left', fontweight='bold')
        ax_dash.text(0.50, 0.51, "GAP (Arc Length)", transform=ax_dash.transAxes, fontsize=9, 
                     color='#888888', ha='center', fontweight='bold')
        ax_dash.text(0.50, 0.22, "POSITION", transform=ax_dash.transAxes, fontsize=9, 
                     color='#888888', ha='center', fontweight='bold')
        
    def _setup_telemetry_bars(self):
        """Setup telemetry bars"""
        ax_bars = self.fig.add_axes([0.785, 0.06, 0.21, 0.24])
        ax_bars.set_facecolor('#0a0a0a')
        ax_bars.set_title('Accel & Input (Follower: Red, Leader: Blue)', fontsize=10, color='white', fontweight='bold', pad=8)
        ax_bars.set_xlim(0, 1)
        ax_bars.set_ylim(-1.0, 1.0)
        ax_bars.set_xticks([])
        ax_bars.set_yticks([-1, -0.5, 0, 0.5, 1.0])
        ax_bars.set_yticklabels(['-100%', '-50%', '0', '+50%', '+100%'], fontsize=8, color='white')
        ax_bars.axhline(0, color='#666666', linewidth=2)
        ax_bars.grid(True, axis='y', alpha=0.2, color='#333333')
        for spine in ax_bars.spines.values():
            spine.set_edgecolor('#555555')
            spine.set_linewidth(2)

        # Bar positions
        bar_x = [0.2, 0.50, 0.80]
        bar_width = 0.15
        bar_half = bar_width / 2
        bar_offset = 0.04

        # Create dual bars
        def make_dual_bar(x_pos, label_text):
            # Follower (left side, red)
            f_pos = ax_bars.bar(x_pos - bar_offset, 0, bar_half, bottom=0, color='#ff3333',
                                edgecolor='white', linewidth=1)[0]
            f_neg = ax_bars.bar(x_pos - bar_offset, 0, bar_half, bottom=0, color='#990000',
                                edgecolor='white', linewidth=1)[0]
            
            # Leader (right side, blue)
            l_pos = ax_bars.bar(x_pos + bar_offset, 0, bar_half, bottom=0, color='#3366ff',
                                edgecolor='white', linewidth=1)[0]
            l_neg = ax_bars.bar(x_pos + bar_offset, 0, bar_half, bottom=0, color='#000099',
                                edgecolor='white', linewidth=1)[0]
            
            # Label
            ax_bars.text(x_pos, -1.15, label_text, ha='center', va='top', fontsize=9, 
                         color='white', fontweight='bold')
            
            return (f_pos, f_neg, l_pos, l_neg)

        # Create all bars
        self.bar_throttle = make_dual_bar(bar_x[0], 'Throttle/\nBrake')
        self.bar_ax = make_dual_bar(bar_x[1], 'Accel X')
        self.bar_ay = make_dual_bar(bar_x[2], 'Accel Y')
        
    def _setup_friction_circles(self):
        """Setup friction circles"""
        ax_fc_front = self.fig.add_axes([0.15, 0.75, 0.20, 0.20])
        ax_fc_rear = self.fig.add_axes([0.15, 0.45, 0.20, 0.20])

        for ax_fc in (ax_fc_front, ax_fc_rear):
            ax_fc.set_facecolor('#0a0a0a')
            ax_fc.grid(True, alpha=0.3, color='#333')
            ax_fc.set_aspect('equal', adjustable='box')
            ax_fc.tick_params(colors='white')
            for sp in ax_fc.spines.values():
                sp.set_edgecolor('#555')

        ax_fc_front.set_title('Front Axle', color='white', fontsize=14, fontweight='bold')
        ax_fc_front.set_xlabel('← Lateral → (N)', color='white')
        ax_fc_front.set_ylabel('← Longitudinal (N) →', color='white')
        ax_fc_rear.set_title('Rear Axle', color='white', fontsize=14, fontweight='bold')
        ax_fc_rear.set_xlabel('← Lateral → (N)', color='white')
        ax_fc_rear.set_ylabel('← Longitudinal (N) →', color='white')

        # Artists to update each frame
        self.front_dot_F, = ax_fc_front.plot([], [], 'o', color='#ff3333', mec='white', mew=1.5, ms=8, label='Follower')
        self.front_dot_L, = ax_fc_front.plot([], [], 'o', color='#3366ff', mec='white', mew=1.5, ms=8, label='Leader')
        ax_fc_front.legend(facecolor='#2a2a2a', edgecolor='#555', labelcolor='white')

        self.rear_dot_F, = ax_fc_rear.plot([], [], 'o', color='#ff3333', mec='white', mew=1.5, ms=8, label='Follower')
        self.rear_dot_L, = ax_fc_rear.plot([], [], 'o', color='#3366ff', mec='white', mew=1.5, ms=8, label='Leader')
        ax_fc_rear.legend(facecolor='#2a2a2a', edgecolor='#555', labelcolor='white')

        # Auto limits from capacity data
        FCX = float(np.nanmax(np.abs([self.data_processor.Fxmax_f_F, self.data_processor.Fxmax_f_L, 
                                      self.data_processor.Fxmax_r_F, self.data_processor.Fxmax_r_L])) + 1e-6)
        FCY = float(np.nanmax(np.abs([self.data_processor.Fymax_f_F, self.data_processor.Fymax_f_L, 
                                      self.data_processor.Fymax_r_F, self.data_processor.Fymax_r_L])) + 1e-6)
        for ax_fc in (ax_fc_front, ax_fc_rear):
            ax_fc.set_xlim(-1.1*FCX, 1.1*FCX)
            ax_fc.set_ylim(-1.1*FCY, 1.1*FCY)
        
    def _on_scroll(self, event):
        """Handle scroll events for zoom"""
        if self.animation is None:
            return
        if event.button == "up":
            self.animation.zoom_level[0] = max(5, self.animation.zoom_level[0] * 0.9)
        elif event.button == "down":
            self.animation.zoom_level[0] = min(200, self.animation.zoom_level[0] * 1.1)
            
    def animate(self):
        """Run the animation"""
        print("Starting animation...")
        
        ani = FuncAnimation(
            self.fig, 
            self._update_all,
            frames=len(self.data_processor.t),
            interval=50,
            blit=True,
            repeat=True,
            cache_frame_data=False
        )
        
        plt.tight_layout()
        plt.show()
        
        return ani
    
    def _update_all(self, frame_idx):
        """Update all dashboard components"""
        data_idx = frame_idx

        # Update main animation
        animation_artists = self.animation.update(frame_idx) if self.animation is not None else []
        
        # Update minimap
        xF, yF, xL, yL = self.precomputed_data.minimap_data[data_idx]
        self.mini_carF.set_data([xF], [yF])
        self.mini_carL.set_data([xL], [yL])

        # Update steering wheel
        spoke_data, center_mark_data, steer_deg = self.precomputed_data.steering_data[data_idx]
        for ln, (x_data, y_data) in zip(self.spokes_lines, spoke_data):
            ln.set_data(x_data, y_data)
        self.center_mark.set_data(*center_mark_data)
        self.steer_text.set_text(f'{steer_deg:+.1f}°')

        # Update GG diagram
        ay_L, ax_L, ay_F, ax_F = self.precomputed_data.gg_data[data_idx]
        self.gg_leader_dot.set_data([ay_L], [ax_L])
        self.gg_follower_dot.set_data([ay_F], [ax_F])

        # Update dashboard
        time_val, spdF_val, spdL_val, gap_s_val, gap_xy_val = self.precomputed_data.dashboard_data[data_idx]
        self.dash_vals["time"].set_text(f"{time_val:05.2f}s")
        self.dash_vals["spdF"].set_text(f"{spdF_val:6.2f} m/s")
        self.dash_vals["spdL"].set_text(f"{spdL_val:6.2f} m/s")
        
        gap_color = '#00ff00' if gap_s_val > 0 else '#ff3333'
        self.dash_vals["gapS"].set_text(f"{gap_s_val:+.2f} m")
        self.dash_vals["gapS"].set_color(gap_color)
        self.dash_vals["gapD"].set_text(f"(Euclidean: {gap_xy_val:.2f}m)")

        if gap_s_val > 0:
            self.dash_vals["p1"].set_text("P1: Follower  P2: Leader")
            self.dash_vals["p1"].set_color('#00ff00')
        else:
            self.dash_vals["p1"].set_text("P1: Leader  P2: Follower")
            self.dash_vals["p1"].set_color('#ffaa00')

        # Update bars
        norm_ax_F, norm_ax_L, norm_ay_F, norm_ay_L = self.precomputed_data.bar_data[data_idx]
        self._update_dual_bar_fast(self.bar_throttle, norm_ax_F, norm_ax_L)
        self._update_dual_bar_fast(self.bar_ax, norm_ax_F, norm_ax_L)
        self._update_dual_bar_fast(self.bar_ay, norm_ay_F, norm_ay_L)

        # Update friction circles
        Ffy_F, Ffx_F, Ffy_L, Ffx_L, Fry_F, Frx_F, Fry_L, Frx_L = self.precomputed_data.friction_data[data_idx]
        self.front_dot_F.set_data([Ffy_F], [Ffx_F])
        self.front_dot_L.set_data([Ffy_L], [Ffx_L])
        self.rear_dot_F.set_data([Fry_F], [Frx_F])
        self.rear_dot_L.set_data([Fry_L], [Frx_L])

        # Combine all artists
        all_artists = animation_artists + [
            self.mini_carF, self.mini_carL, *self.spokes_lines, self.center_mark, self.steer_text,
            self.gg_leader_dot, self.gg_follower_dot, *self.dash_vals.values(),
            *self.bar_throttle, *self.bar_ax, *self.bar_ay,
            self.front_dot_F, self.front_dot_L, self.rear_dot_F, self.rear_dot_L
        ]
        
        return all_artists

    def _update_dual_bar_fast(self, bars, f_val, l_val):
        """Update dual bar display"""
        f_pos, f_neg, l_pos, l_neg = bars
        
        # Follower
        if f_val >= 0:
            f_pos.set_height(f_val)
            f_pos.set_y(0)
            f_neg.set_height(0)
        else:
            f_neg.set_height(-f_val)
            f_neg.set_y(f_val)
            f_pos.set_height(0)
        
        # Leader
        if l_val >= 0:
            l_pos.set_height(l_val)
            l_pos.set_y(0)
            l_neg.set_height(0)
        else:
            l_neg.set_height(-l_val)
            l_neg.set_y(l_val)
            l_pos.set_height(0)


class PrecomputedData:
    """Precomputes all animation data for performance"""
    
    def __init__(self, data_processor):
        self.dp = data_processor
        self.carF_vertices = []
        self.carL_vertices = []
        self.steering_data = []
        self.bar_data = []
        self.gg_data = []
        self.dashboard_data = []
        self.minimap_data = []
        self.friction_data = []
        
    def precompute_all(self):
        """Precompute all animation data"""
        indices = range(len(self.dp.t))
        
        for idx in indices:
            i = idx
            # Car positions
            carXcoords = np.array([-self.dp.b_m, -self.dp.b_m, self.dp.a_m, self.dp.a_m])
            carWidth = 1.8
            carYcoords = np.array([-carWidth/2, carWidth/2, carWidth/2, -carWidth/2])
            
            xRotF = self.dp.xF_interp[i] + np.cos(self.dp.carAngleF_interp[i]) * carXcoords - np.sin(self.dp.carAngleF_interp[i]) * carYcoords
            yRotF = self.dp.yF_interp[i] + np.sin(self.dp.carAngleF_interp[i]) * carXcoords + np.cos(self.dp.carAngleF_interp[i]) * carYcoords
            self.carF_vertices.append(np.column_stack([xRotF, yRotF]))
            
            xRotL = self.dp.xL_interp[i] + np.cos(self.dp.carAngleL_interp[i]) * carXcoords - np.sin(self.dp.carAngleL_interp[i]) * carYcoords
            yRotL = self.dp.yL_interp[i] + np.sin(self.dp.carAngleL_interp[i]) * carXcoords + np.cos(self.dp.carAngleL_interp[i]) * carYcoords
            self.carL_vertices.append(np.column_stack([xRotL, yRotL]))

            # Steering wheel
            wheel_deg = np.clip(self.dp.STEER_VISUAL_SIGN * self.dp.STEER_WHEEL_DEG_PER_RAD * self.dp.deltaF_i[i] + self.dp.STEER_ZERO_OFFSET_DEG, -540.0, 540.0)
            ang = np.deg2rad(wheel_deg)
            c, s = np.cos(ang), np.sin(ang)
            
            spoke_data = []
            for base_ang in self.dp.base_spokes:
                x1b = self.dp.spoke_len * np.cos(base_ang)
                y1b = self.dp.spoke_len * np.sin(base_ang)
                x1 = c * x1b - s * y1b
                y1 = s * x1b + c * y1b
                spoke_data.append(([0.0, x1], [0.0, y1]))
            
            mark_x = self.dp.spoke_len * 0.30 * np.sin(ang)
            mark_y = self.dp.spoke_len * 0.30 * np.cos(ang)
            center_mark_data = ([mark_x*0.7, mark_x*1.1], [mark_y*0.7, mark_y*1.1])
            
            self.steering_data.append((spoke_data, center_mark_data, np.degrees(self.dp.deltaF_i[i])))

            # Bar data
            norm_ax_F = np.clip(self.dp.ax_F_interp[i] / self.dp.AX_LIM, -1.0, 1.0)
            norm_ax_L = np.clip(self.dp.ax_L_interp[i] / self.dp.AX_LIM, -1.0, 1.0)
            norm_ay_F = np.clip(self.dp.ay_F_interp[i] / self.dp.AY_LIM, -1.0, 1.0)
            norm_ay_L = np.clip(self.dp.ay_L_interp[i] / self.dp.AY_LIM, -1.0, 1.0)
            self.bar_data.append((norm_ax_F, norm_ax_L, norm_ay_F, norm_ay_L))

            # GG diagram data
            self.gg_data.append((
                self.dp.ay_L_direct_interp[i], self.dp.ax_L_direct_interp[i], # Leader direct
                self.dp.ay_F_direct_interp[i], self.dp.ax_F_direct_interp[i]  # Follower direct
            ))

            # Dashboard data
            self.dashboard_data.append((
                self.dp.t[i], self.dp.spdF[i], self.dp.spdL[i], self.dp.gap_s[i], self.dp.gap_xy[i]
            ))

            # Minimap data
            self.minimap_data.append((self.dp.xF_interp[i], self.dp.yF_interp[i], self.dp.xL_interp[i], self.dp.yL_interp[i]))

            # Friction circle data
            self.friction_data.append((
                self.dp.Ffy_F[i], self.dp.Ffx_F[i], self.dp.Ffy_L[i], self.dp.Ffx_L[i],
                self.dp.Fry_F[i], self.dp.Frx_F[i], self.dp.Fry_L[i], self.dp.Frx_L[i]
            ))

        print("Pre-computation complete!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the Matplotlib-only telemetry HUD.")
    parser.add_argument("leader_file", help="Leader trajectory .mat file")
    parser.add_argument("follower_file", help="Follower trajectory .mat file")
    parser.add_argument(
        "track_file",
        nargs="?",
        default="NASCAR_Track_Monge_v3.mat",
        help="Track .mat file, default: NASCAR_Track_Monge_v3.mat",
    )
    args = parser.parse_args()

    dashboard = TelemetryDashboard(args.leader_file, args.follower_file, args.track_file)
    dashboard.setup_dashboard()
    ani = dashboard.animate()
