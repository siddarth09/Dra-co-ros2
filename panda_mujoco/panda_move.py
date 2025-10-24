#!/usr/bin/env python3
"""
panda_move.py — Unified MuJoCo + Drake Panda Controller

Provides:
- MuJoCo simulation handling
- Drake IK solver interface
- Optional Meshcat visualization
"""

import time
import numpy as np
import mujoco as mu
from mujoco import viewer
from pydrake.all import (
    MultibodyPlant, Parser, InverseKinematics,
    RigidTransform, RotationMatrix, StartMeshcat,
    Simulator, RobotDiagramBuilder
)
from pydrake.solvers import Solve
from manipulation.make_drake_compatible_model import MakeDrakeCompatibleModel
from manipulation.remotes import AddMujocoMenagerie
from manipulation.utils import ApplyDefaultVisualization


class PandaMove:
    def __init__(self, xml_path: str, visualization: bool = True):
        """Initialize both MuJoCo and Drake models."""
        self.xml_path = xml_path
        self.visualize = visualization

        self._load_mujoco_model()
        self._setup_drake_model()
        if self.visualize:
            self._setup_meshcat()

    # ------------------------------------------------------------------
    # INITIALIZATION
    # ------------------------------------------------------------------

    def _load_mujoco_model(self):
        """Load the MuJoCo model for simulation."""
        print(f" Loading MuJoCo model: {self.xml_path}")
        self.model = mu.MjModel.from_xml_path(self.xml_path)
        self.data = mu.MjData(self.model)
        print(f" Model loaded with {self.model.njnt} joints.")

    def _setup_drake_model(self):
        """Convert the MuJoCo XML to Drake-compatible XML and load it."""
        print(" Converting model for Drake...")
        drake_xml = self.xml_path.replace(".xml", ".drake.xml")
        MakeDrakeCompatibleModel(self.xml_path, drake_xml, overwrite=True)

        builder = RobotDiagramBuilder()
        plant = builder.plant()
        parser = Parser(plant)
        AddMujocoMenagerie(parser.package_map())
        parser.AddModels(drake_xml)
        plant.Finalize()

        self.plant = plant
        self.builder = builder
        self.ee_frame = self._detect_end_effector()
        print(f" End-effector frame: {self.ee_frame.name()}")

    def _setup_meshcat(self):
        """Start Meshcat visualization."""
        print(" Starting Meshcat...")
        self.meshcat = StartMeshcat()
        ApplyDefaultVisualization(self.builder.builder(), meshcat=self.meshcat)
        self.diagram = self.builder.Build()
        self.simulator = Simulator(self.diagram)
        self.context = self.simulator.get_mutable_context()

    # ------------------------------------------------------------------
    # UTILITIES
    # ------------------------------------------------------------------

    def _detect_end_effector(self):
        """Try to find the EE frame automatically."""
        for name in ["hand", "link7", "panda_link8"]:
            try:
                return self.plant.GetFrameByName(name)
            except RuntimeError:
                continue
        raise RuntimeError(" Could not find a valid end-effector frame.")

    def get_current_joint_positions(self):
        """Get current joint positions from MuJoCo."""
        return np.copy(self.data.qpos)
    
    def get_ee_pose(self):
        context= self.plant.CreateDefaultContext()
        X_WE = self.ee_frame.CalcPoseInWorld(context)
        
        return X_WE.rotation(), X_WE.translation()
    
    def change_targets(self, targets):
        return RigidTransform(
            RotationMatrix.MakeZRotation(np.pi / 2),
            targets
        )

    def _interp_positions(self, p0, p1, steps):
        """Linear interpolation between two positions."""
        p0= np.asarray(p0)
        p1= np.asarray(p1)
        
        alphas = np.linspace(0.0,1.0,steps)

        return [(1-a)*p0 + a*p1 for a in alphas]
    
    
    # ------------------------------------------------------------------
    # MOTION + IK
    # ------------------------------------------------------------------

    def solve_ik(self, target_pose: RigidTransform):
        """Solve IK for the target pose and return joint configuration."""
        ik = InverseKinematics(self.plant)
        q = ik.q()

        ik.AddPositionConstraint(
            self.ee_frame, [0, 0, 0],
            self.plant.world_frame(),
            target_pose.translation() - [0.001]*3,
            target_pose.translation() + [0.001]*3,
        )
        ik.AddOrientationConstraint(
            frameAbar=self.ee_frame,
            R_AbarA=RotationMatrix(),
            frameBbar=self.plant.world_frame(),
            R_BbarB=target_pose.rotation(),
            theta_bound=0.01,
        )

        result = Solve(ik.prog())
        if not result.is_success():
            raise RuntimeError("IK solution not found.")

        return result.GetSolution(q)
    
    
    def plan_cubic_joint_trajectory(self,q_start,q_end,steps=150):
        """Smooth time parameterized cubic joint trajectory."""
        q_start = np.asarray(q_start)
        q_end = np.asarray(q_end)
        n = len(q_start)
        
        v0 = np.zeros(n)
        vf= np.zeros(n)
        
        T = 1.0  # normalized time duration
        t = np.linspace(0.0,T,steps)
        q_traj = [] 
        qd_traj = []
        for ti in t :
            a0 = q_start 
            a1 = v0 
            a2 = 3 * (q_end - q_start) - 2 * v0 - vf 
            a3 = -2 * (q_end - q_start) + v0 + vf 
            q = a0 + a1*ti + a2*ti**2 + a3*ti**3
            
            qd= a1 + 2*a2*ti + 3*a3*ti**2
            q_traj.append(q)
            qd_traj.append(qd)
            
        return np.array(q_traj), np.array(qd_traj)

    def run_sequence(self, targets, duration=2.0):
        with viewer.launch_passive(self.model, self.data) as v:
            q_traj,_ = self.plan_cubic_joint_trajectory(targets[0], targets[1], steps=150)
            for i in range(len(targets)):
                q_start = self.get_current_joint_positions()
                q_end = np.copy(targets[i])
                n_steps = int(duration / 0.01)

                q_traj, _ = self.plan_cubic_joint_trajectory(q_start, q_end, steps=n_steps)

                for q in q_traj:
                    self.data.qpos[:] = q
                    mu.mj_forward(self.model, self.data)
                    mu.mj_step(self.model, self.data)
                    v.sync()
                    time.sleep(0.01)
            
                print(" Reached one target — moving to next")
            print(" All motions done! Keeping viewer open.")
            while v.is_running():
                v.sync()
                time.sleep(0.01)