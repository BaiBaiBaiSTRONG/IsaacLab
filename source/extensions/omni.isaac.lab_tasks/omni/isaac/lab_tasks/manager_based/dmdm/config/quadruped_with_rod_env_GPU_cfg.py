
import numpy as np
import torch


import omni.isaac.core.utils.prims as prim_utils

#import omni.isaac.lab.envs.mdp as mdp


import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.assets import Articulation, ArticulationCfg, AssetBaseCfg, RigidObjectCfg, RigidObject
from omni.isaac.lab.envs import ManagerBasedEnv, ManagerBasedEnvCfg
from omni.isaac.lab_tasks.utils.wrappers.rsl_rl import RslRlPpoActorCriticCfg
from omni.isaac.lab.managers import EventTermCfg as EventTerm
from omni.isaac.lab.managers import ObservationGroupCfg as ObsGroup
from omni.isaac.lab.managers import ObservationTermCfg as ObsTerm
from omni.isaac.lab.managers import RewardTermCfg as RewTerm
from omni.isaac.lab.managers import SceneEntityCfg
from omni.isaac.lab.managers import TerminationTermCfg as DoneTerm
from omni.isaac.lab.utils.noise import AdditiveUniformNoiseCfg as Unoise
from omni.isaac.lab.scene import InteractiveSceneCfg, InteractiveScene
from omni.isaac.lab.sensors import ContactSensorCfg, ImuCfg, RayCasterCfg, patterns
from omni.isaac.lab.sim import PhysxCfg
from omni.isaac.lab.terrains import TerrainImporterCfg
from omni.isaac.lab.utils import configclass


from omni.isaac.lab_tasks.manager_based.dmdm import mdp
##
# Pre-defined configs
##
#from omni.isaac.lab_assets.anymal import ANYMAL_B_CFG, ANYMAL_C_CFG, ANYMAL_D_CFG  # isort:skip
#from omni.isaac.lab_assets.spot import SPOT_CFG  # isort:skip
#from omni.isaac.lab_assets.unitree import UNITREE_GO2_CFG  # isort:skip
#from omni.isaac.lab_assets.unitree import UNITREE_RODROPEGO2_nocollider_CFG  # isort:skip
#from unitree import UNITREE_RODROPEGO2_CFG
from omni.isaac.lab_tasks.manager_based.dmdm.env.unitree import UNITREE_RODROPEGO2_CFG, UNITREE_GO1_CFG, UNITREE_GO2_CFG

# Scene definition
@configclass
class DMDMSceneCfg(InteractiveSceneCfg):

    # Add terrain
    terrain = AssetBaseCfg(
        prim_path="/World/ground",
        spawn=sim_utils.GroundPlaneCfg()
    )
    #prim_utils.create_prim("/World/envs/env_.*/object", "Xform")
    

    # Add robot
    robot: ArticulationCfg = UNITREE_RODROPEGO2_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot", 
                                                            init_state=ArticulationCfg.InitialStateCfg(pos=(0.0, 0.0, 0.32)))
    #robot_data = Articulation(ArticulationCfg)
    # for n in range(10):
    #     # prim_utils.create_prim(f"/World/envs/env_{i}/Object", "Xform")
    #     print("wocaosinidema")

    target: RigidObjectCfg = RigidObjectCfg(prim_path="{ENV_REGEX_NS}/Object", 
                                spawn=sim_utils.UsdFileCfg(usd_path="/home/hanyang/Dependecies/IsaacLab/source/extensions/omni.isaac.lab_tasks/omni/isaac/lab_tasks/manager_based/dmdm/env/Models/target_b.usd",
                                                           rigid_props=sim_utils.RigidBodyPropertiesCfg(rigid_body_enabled=True, disable_gravity=False, kinematic_enabled=True),
                                                           collision_props=sim_utils.CollisionPropertiesCfg(),
                                                           mass_props=sim_utils.MassPropertiesCfg(mass=1.0, density=10.0)
                                                           ),
                                init_state=RigidObjectCfg.InitialStateCfg(pos=(1.3, 0.0, 0.8)))

    # Sensors
    # IMU
    imu_robot = ImuCfg(prim_path="{ENV_REGEX_NS}/Robot/imu")
        # Deactivated because of the USD Design Error. (Dec 29 2024)
    # imu_FL = ImuCfg(prim_path="{ENV_REGEX_NS}/Robot/base/imu_FL")
    # imu_FR = ImuCfg(prim_path="{ENV_REGEX_NS}/Robot/base/imu_FR")
    # imu_RL = ImuCfg(prim_path="{ENV_REGEX_NS}/Robot/base/imu_RL")
    # imu_RR = ImuCfg(prim_path="{ENV_REGEX_NS}/Robot/base/imu_RR")
    imu_end_ball = ImuCfg(prim_path="{ENV_REGEX_NS}/Robot/end_ball")
    # Height scanner
    height_scanner = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base",
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)),
        attach_yaw_only=True,
        pattern_cfg=patterns.GridPatternCfg(resolution=0.1, size=(1.0, 0.8)),
        debug_vis=True,
        mesh_prim_paths=["/World/ground"],
    )
    # Contact sensors
    contact_forces = ContactSensorCfg(prim_path="{ENV_REGEX_NS}/Robot/.*", history_length=3, track_air_time=True)
    contact_forces_FL = ContactSensorCfg(prim_path="{ENV_REGEX_NS}/Robot/FL_foot", update_period=0.0, history_length=3, track_air_time=True)
    contact_forces_FR = ContactSensorCfg(prim_path="{ENV_REGEX_NS}/Robot/FR_foot", update_period=0.0, history_length=3, track_air_time=True)
    contact_forces_RL = ContactSensorCfg(prim_path="{ENV_REGEX_NS}/Robot/RL_foot", update_period=0.0, history_length=3, track_air_time=True)
    contact_forces_RR = ContactSensorCfg(prim_path="{ENV_REGEX_NS}/Robot/RR_foot", update_period=0.0, history_length=3, track_air_time=True)
    contact_forces_base = ContactSensorCfg(prim_path="{ENV_REGEX_NS}/Robot/base", update_period=0.0, history_length=3, track_air_time=True)
    contact_forces_endball = ContactSensorCfg(prim_path="{ENV_REGEX_NS}/Robot/end_ball", update_period=0.0, history_length=3, track_air_time=True)
    contact_forces_targetbox = ContactSensorCfg(prim_path="{ENV_REGEX_NS}/Object", update_period=0.0, history_length=3, track_air_time=True)

    # Lights
    dome_light = AssetBaseCfg(
        prim_path="/World/DomeLight",
        spawn=sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
    )

@configclass
class ActionsCfg:
    joint_pos = mdp.JointPositionActionCfg(asset_name="robot", joint_names=['FL_hip_joint', 'FR_hip_joint', 'RL_hip_joint', 'RR_hip_joint', 'FL_thigh_joint', 'FR_thigh_joint', 'RL_thigh_joint', 'RR_thigh_joint', 'FL_calf_joint', 'FR_calf_joint', 'RL_calf_joint', 'RR_calf_joint'], scale=0.8)
    rod_pos = mdp.JointPositionActionCfg(asset_name="robot", joint_names=["rod_motor_position", "base_motor_velocity"], scale=1.0) #Fan de
    #base_vel = mdp.JointVelocityActionCfg(asset_name="robot", joint_names=["base_motor_velocity"], scale=1.0)
    #d6joint_pos = mdp.JointPositionActionCfg(asset_name="robot", joint_names=["D6Joint.*"], scale=1.0)

@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        #velocity_commands = ObsTerm(func=constant_commands)
        joint_pos = ObsTerm(func=mdp.joint_pos_rel, noise=Unoise(n_min=-0.01, n_max=0.01))
        joint_vel = ObsTerm(func=mdp.joint_vel_rel, noise=Unoise(n_min=-1.5, n_max=1.5))
        
        rod_motor_pos = ObsTerm(func=mdp.rod_motor_position, noise=Unoise(n_min=-0.02, n_max=0.02))
        base_motor_pos = ObsTerm(func=mdp.base_motor_position, noise=Unoise(n_min=-0.02, n_max=0.02))
        

        actions = ObsTerm(func=mdp.last_action)
        

        imu_orientation_robot = ObsTerm(func=mdp.imu_orientation, params={"sensor_cfg": SceneEntityCfg("imu_robot")}, noise=Unoise(n_min=-0.05, n_max=0.05))
        target_pos = ObsTerm(func=mdp.root_pos_w, params={"asset_cfg": SceneEntityCfg("target")}, noise=Unoise(n_min=-0.01, n_max=0.01))


        # '''Introduced from Actor Critic  part , we block this here to directly use basic train.
        # '''
        # height_scan = ObsTerm(
        #     func=mdp.height_scan,
        #     params={"sensor_cfg": SceneEntityCfg("height_scanner"), "offset": 0.0},
        #     noise=Unoise(n_min=-0.1, n_max=0.1),
        #     clip=(-1.0, 1.0),
        # )

        # Ball information all the data from the IMU sensor built inside the ball

        # end_ball_pos = ObsTerm(func=mdp.imu_position, params={"sensor_cfg": SceneEntityCfg("imu_end_ball")}, noise=Unoise(n_min=-0.1, n_max=0.1))
        # end_ball_lin_vel = ObsTerm(func=mdp.imu_lin_vel, params={"sensor_cfg": SceneEntityCfg("imu_end_ball")}, noise=Unoise(n_min=-0.1, n_max=0.1))
        # end_ball_lin_acc = ObsTerm(func=mdp.imu_lin_acc, params={"sensor_cfg": SceneEntityCfg("imu_end_ball")}, noise=Unoise(n_min=-0.1, n_max=0.1))
        # end_ball_ang_vel = ObsTerm(func=mdp.imu_ang_vel, params={"sensor_cfg": SceneEntityCfg("imu_end_ball")}, noise=Unoise(n_min=-0.2, n_max=0.2))
        # end_ball_ang_acc = ObsTerm(func=mdp.imu_ang_acc, params={"sensor_cfg": SceneEntityCfg("imu_end_ball")}, noise=Unoise(n_min=-0.2, n_max=0.2))
        # end_ball_imu_orientation = ObsTerm(func=mdp.imu_orientation, params={"sensor_cfg": SceneEntityCfg("imu_end_ball")}, noise=Unoise(n_min=-0.05, n_max=0.05))
      
        # # Feet position
        # contact_FL_pos = ObsTerm(func=mdp.contact_sensor_pos, params={"sensor_cfg": SceneEntityCfg("contact_forces_FL")}, noise=Unoise(n_min=-0.01, n_max=0.01))
        # contact_FR_pos = ObsTerm(func=mdp.contact_sensor_pos, params={"sensor_cfg": SceneEntityCfg("contact_forces_FR")}, noise=Unoise(n_min=-0.01, n_max=0.01))
        # contact_RL_pos = ObsTerm(func=mdp.contact_sensor_pos, params={"sensor_cfg": SceneEntityCfg("contact_forces_RL")}, noise=Unoise(n_min=-0.01, n_max=0.01))
        # contact_RR_pos = ObsTerm(func=mdp.contact_sensor_pos, params={"sensor_cfg": SceneEntityCfg("contact_forces_RR")}, noise=Unoise(n_min=-0.01, n_max=0.01))

        # Four feet contact point real-time position，based on the contact sensor
        # FL means the left front foot, FR means the right front foot, RL means the left rear(back) foot, RR means the right rear(back) foot.
        # FL_real_time_position = ObsTerm(func=mdp.feet_real_time_position, params={"feet_contact_sensor_cfg": SceneEntityCfg("contact_forces_FL"), "offset": 0.0}, noise=Unoise(n_min=-0.1, n_max=0.1))
        # FR_real_time_position = ObsTerm(func=mdp.feet_real_time_position, params={"feet_contact_sensor_cfg": SceneEntityCfg("contact_forces_FR"), "offset": 0.0}, noise=Unoise(n_min=-0.1, n_max=0.1))
        # RL_real_time_position = ObsTerm(func=mdp.feet_real_time_position, params={"feet_contact_sensor_cfg": SceneEntityCfg("contact_forces_RL"), "offset": 0.0}, noise=Unoise(n_min=-0.1, n_max=0.1))
        # RR_real_time_position = ObsTerm(func=mdp.feet_real_time_position, params={"feet_contact_sensor_cfg": SceneEntityCfg("contact_forces_RR"), "offset": 0.0}, noise=Unoise(n_min=-0.1, n_max=0.1))
        

        

        def __post_init__(self):
            """Whether to enable corruption for the observation group. Defaults to False.

            If true, the observation terms in the group are corrupted by adding noise (if specified).
            Otherwise, no corruption is applied.
            """
            self.enable_corruption = True



            #TODO: check if this is necessary, doc said if this is False, the terms in the group will be concatenated
            '''Whether to concatenate the observation terms in the group. Defaults to True.
            If true, the observation terms in the group are concatenated along the last dimension.
            Otherwise, they are kept separate and returned as a dictionary.
            If the observation group contains terms of different dimensions, it must be set to False.
            '''
            self.concatenate_terms = True   # False
    
    





    @configclass
    class CriticCfg(ObsGroup):
        """Observations for critic group."""

        

        # height_scan = ObsTerm(
        #     func=mdp.height_scan,
        #     params={"sensor_cfg": SceneEntityCfg("height_scanner"), "offset": 0.0},
        #     noise=Unoise(n_min=-0.1, n_max=0.1),
        #     clip=(-1.0, 1.0),
        # )

        # observation terms (order preserved)
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel, noise=Unoise(n_min=-0.1, n_max=0.1))
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, noise=Unoise(n_min=-0.2, n_max=0.2))
        #root_pos_w = ObsTerm(func=mdp.root_pos_w, noise=Unoise(n_min=-0.1, n_max=0.1))
        projected_gravity = ObsTerm(
            func=mdp.projected_gravity,
            noise=Unoise(n_min=-0.05, n_max=0.05),
        )



        # Ball information all the data from the IMU sensor built inside the ball

        end_ball_pos = ObsTerm(func=mdp.imu_position, params={"sensor_cfg": SceneEntityCfg("imu_end_ball")}, noise=Unoise(n_min=-0.1, n_max=0.1))
        end_ball_lin_vel = ObsTerm(func=mdp.imu_lin_vel, params={"sensor_cfg": SceneEntityCfg("imu_end_ball")}, noise=Unoise(n_min=-0.1, n_max=0.1))
        end_ball_lin_acc = ObsTerm(func=mdp.imu_lin_acc, params={"sensor_cfg": SceneEntityCfg("imu_end_ball")}, noise=Unoise(n_min=-0.1, n_max=0.1))
        end_ball_ang_vel = ObsTerm(func=mdp.imu_ang_vel, params={"sensor_cfg": SceneEntityCfg("imu_end_ball")}, noise=Unoise(n_min=-0.2, n_max=0.2))
        end_ball_ang_acc = ObsTerm(func=mdp.imu_ang_acc, params={"sensor_cfg": SceneEntityCfg("imu_end_ball")}, noise=Unoise(n_min=-0.2, n_max=0.2))
        
        # NOT ACTIVATED BECAUSE OF THE Unknown IsaacLab DESIGN ERROR. (Dec 29 2024)
        # end_ball_imu_orientation = ObsTerm(func=mdp.imu_orientation, params={"sensor_cfg": SceneEntityCfg("imu_end_ball")}, noise=Unoise(n_min=-0.05, n_max=0.05))



      
        # Four feet contact point real-time position，based on the contact sensor
        # FL means the left front foot, FR means the right front foot, RL means the left rear(back) foot, RR means the right rear(back) foot.

        # IF YOU NEED THIS, MUST switch the config class ContactSensorCfg(SensorBaseCfg) -> track_pose: bool = True.  (Dec 29 2024)
        contact_FL_pos = ObsTerm(func=mdp.contact_sensor_pos, params={"sensor_cfg": SceneEntityCfg("contact_forces_FL")}, noise=Unoise(n_min=-0.01, n_max=0.01))
        contact_FR_pos = ObsTerm(func=mdp.contact_sensor_pos, params={"sensor_cfg": SceneEntityCfg("contact_forces_FR")}, noise=Unoise(n_min=-0.01, n_max=0.01))
        contact_RL_pos = ObsTerm(func=mdp.contact_sensor_pos, params={"sensor_cfg": SceneEntityCfg("contact_forces_RL")}, noise=Unoise(n_min=-0.01, n_max=0.01))
        contact_RR_pos = ObsTerm(func=mdp.contact_sensor_pos, params={"sensor_cfg": SceneEntityCfg("contact_forces_RR")}, noise=Unoise(n_min=-0.01, n_max=0.01))



        # Four Corner Information of the body acuire from the four corner imu sensor, Not Activated because of the USD Design error. (Dec 29 2024)
            
        #     #Corner Position  
        # FL_corner_pos = ObsTerm(func=mdp.imu_position, params={"sensor_cfg": SceneEntityCfg("imu_FL")}, noise=Unoise(n_min=-0.1, n_max=0.1))
        # FR_corner_pos = ObsTerm(func=mdp.imu_position, params={"sensor_cfg": SceneEntityCfg("imu_FR")}, noise=Unoise(n_min=-0.1, n_max=0.1))
        # RL_corner_pos = ObsTerm(func=mdp.imu_position, params={"sensor_cfg": SceneEntityCfg("imu_RL")}, noise=Unoise(n_min=-0.1, n_max=0.1))
        # RR_corner_pos = ObsTerm(func=mdp.imu_position, params={"sensor_cfg": SceneEntityCfg("imu_RR")}, noise=Unoise(n_min=-0.1, n_max=0.1))

        #     #Corner Linear Velocity
        # FL_corner_lin_vel = ObsTerm(func=mdp.imu_lin_vel, params={"sensor_cfg": SceneEntityCfg("imu_FL")}, noise=Unoise(n_min=-0.1, n_max=0.1))
        # FR_corner_lin_vel = ObsTerm(func=mdp.imu_lin_vel, params={"sensor_cfg": SceneEntityCfg("imu_FR")}, noise=Unoise(n_min=-0.1, n_max=0.1))
        # RL_corner_lin_vel = ObsTerm(func=mdp.imu_lin_vel, params={"sensor_cfg": SceneEntityCfg("imu_RL")}, noise=Unoise(n_min=-0.1, n_max=0.1))
        # RR_corner_lin_vel = ObsTerm(func=mdp.imu_lin_vel, params={"sensor_cfg": SceneEntityCfg("imu_RR")}, noise=Unoise(n_min=-0.1, n_max=0.1))

        #     #Corner Linear Acceleration
        # FL_corner_lin_acc = ObsTerm(func=mdp.imu_lin_acc, params={"sensor_cfg": SceneEntityCfg("imu_FL")}, noise=Unoise(n_min=-0.1, n_max=0.1))
        # FR_corner_lin_acc = ObsTerm(func=mdp.imu_lin_acc, params={"sensor_cfg": SceneEntityCfg("imu_FR")}, noise=Unoise(n_min=-0.1, n_max=0.1))
        # RL_corner_lin_acc = ObsTerm(func=mdp.imu_lin_acc, params={"sensor_cfg": SceneEntityCfg("imu_RL")}, noise=Unoise(n_min=-0.1, n_max=0.1))
        # RR_corner_lin_acc = ObsTerm(func=mdp.imu_lin_acc, params={"sensor_cfg": SceneEntityCfg("imu_RR")}, noise=Unoise(n_min=-0.1, n_max=0.1))

        #     #Corner Angular Velocity
        # FL_corner_ang_vel = ObsTerm(func=mdp.imu_ang_vel, params={"sensor_cfg": SceneEntityCfg("imu_FL")}, noise=Unoise(n_min=-0.1, n_max=0.1))
        # FR_corner_ang_vel = ObsTerm(func=mdp.imu_ang_vel, params={"sensor_cfg": SceneEntityCfg("imu_FR")}, noise=Unoise(n_min=-0.1, n_max=0.1))
        # RL_corner_ang_vel = ObsTerm(func=mdp.imu_ang_vel, params={"sensor_cfg": SceneEntityCfg("imu_RL")}, noise=Unoise(n_min=-0.1, n_max=0.1))
        # RR_corner_ang_vel = ObsTerm(func=mdp.imu_ang_vel, params={"sensor_cfg": SceneEntityCfg("imu_RR")}, noise=Unoise(n_min=-0.1, n_max=0.1))

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    # observation groups
    policy: PolicyCfg = PolicyCfg()
    critic: CriticCfg = CriticCfg()




@configclass
class RewardsCfg:
    ''' Basic Reward on Quadruped Robot
    '''

    # Base height reward we want the robot to be as close as possible to the target height
    base_height_follow = RewTerm(func=mdp.base_height_l2, weight=0.003, params={"asset_cfg": SceneEntityCfg("robot"), "target_height": 0.36})
    # Base velocity reward we want the robot to be as close as possible to the target velocity
    x_velocity_follow = RewTerm(func=mdp.track_desired_lin_vel_x_exp, weight=-0.05, params={"std": 0.1, "target_lin_vel_x": 0.0})
    y_velocity_follow = RewTerm(func=mdp.track_desired_lin_vel_y_exp, weight=-0.05, params={"std": 0.1, "target_lin_vel_y": 0.0})
    
    lin_vel_z_l2 = RewTerm(func=mdp.lin_vel_z_l2, weight=-0.001)
    ang_vel_xy_l2 = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.05)
    front_joint_vel_l1 = RewTerm(func=mdp.joint_vel_l2, weight=0.001, params={"asset_cfg": SceneEntityCfg("robot", joint_names=['FL_hip_joint', 'FR_hip_joint', 'FL_thigh_joint', 'FR_thigh_joint', 'RL_thigh_joint', 'FL_calf_joint', 'FR_calf_joint'])})
    back_joint_vel_l1 = RewTerm(func=mdp.joint_vel_l2, weight=-0.0001, params={"asset_cfg": SceneEntityCfg("robot", joint_names=['RL_hip_joint', 'RR_hip_joint', 'RL_thigh_joint', 'RR_thigh_joint', 'RL_calf_joint', 'RR_calf_joint'])})
    dof_go2_torques_l2 = RewTerm(func=mdp.joint_torques_l2, weight=-1.0e-5, params={"asset_cfg": SceneEntityCfg("robot", joint_names=['FL_hip_joint', 'FR_hip_joint', 'RL_hip_joint', 'RR_hip_joint', 'FL_thigh_joint', 'FR_thigh_joint', 'RL_thigh_joint', 'RR_thigh_joint', 'FL_calf_joint', 'FR_calf_joint', 'RL_calf_joint', 'RR_calf_joint'])})
    #feet_air_time = RewTerm(func=mdp.feet_air_time, weight=0.125, params={"sensor_cfg": SceneEntityCfg("contact_foorces", body_names=".*_foot")})
    #flat_orientation_l2 = RewTerm(func=mdp.flat_orientation_l2, weight=0.1)
    
    


    '''DMDM Specific reward added
    '''
    dof_rod_torques_l2 = RewTerm(func=mdp.joint_torques_l2, weight=-3.0e-5, params={"asset_cfg": SceneEntityCfg("robot", joint_names=['rod_motor_position', 'base_motor_velocity'])})

    #rod_motor_target_position = RewTerm(func=mdp.joint_target_position, weight=-2.5, params={"asset_cfg": SceneEntityCfg("robot", joint_names=['rod_motor_position']), "target_position": 0.0})
    #base_motor_joint_target_position = RewTerm(func=mdp.joint_target_position, weight=-2.5, params={"asset_cfg": SceneEntityCfg("robot", joint_names=['base_motor_velocity']), "target_position": 0.0})
    rod_motor_joint_vel = RewTerm(func=mdp.joint_vel_l2, weight= 0.05, params={"asset_cfg": SceneEntityCfg("robot", joint_names=["rod_motor_position"])})
    base_motor_joint_vel = RewTerm(func=mdp.joint_vel_l2, weight=0.05, params={"asset_cfg": SceneEntityCfg("robot", joint_names=["base_motor_velocity"])})


    rod_motor_joint_velocity_too_fast = RewTerm(func=mdp.joint_velocity_too_fast, weight=-0.5, params={"asset_cfg": SceneEntityCfg("robot", joint_names=["rod_motor_position"]), "threshold": 3})
    rod_motor_joint_direction = RewTerm(func=mdp.rod_joint_direction, weight=0.1, params={"scalefactor": 3.0,"asset_cfg": SceneEntityCfg("robot", joint_names=["rod_motor_position"])})
    # base_motor_joint_direction = RewTerm(func=mdp.base_joint_direction, weight=-0.1, params={"asset_cfg": SceneEntityCfg("robot", joint_names=["base_motor_velocity"])})
    
    
    #target_contact = RewTerm(func=mdp.ball_contact_with_box, weight=100.0, params={"threshold": 1.0 ,"end_ball_contact_sensor_config": SceneEntityCfg("contact_forces_endball"), "target_box_contact_sensor_config": SceneEntityCfg("contact_forces_targetbox")})
    FL_feet_air_time = RewTerm(func=mdp.feet_air_time_2, weight=-0.1, params={"sensor_cfg": SceneEntityCfg("contact_forces_RL"), "threshold": 0.7})
    FR_feet_air_time = RewTerm(func=mdp.feet_air_time_2, weight=-0.1, params={"sensor_cfg": SceneEntityCfg("contact_forces_RR"), "threshold": 0.7})
    
    
    target_contact = RewTerm(func=mdp.ball_contact_with_box, weight=50.0, params={"threshold": 0 ,"end_ball_contact_sensor_config": SceneEntityCfg("contact_forces_endball"), "target_box_contact_sensor_config": SceneEntityCfg("contact_forces_targetbox")})
    ball_fly_velocity_direction = RewTerm(func=mdp.ball_fly_velocity_direction, weight=1, params={"scale_factor": 3, "imu_asset_cfg": SceneEntityCfg("imu_end_ball"),"robot_asset_cfg": SceneEntityCfg("robot"),"target_asset_cfg": SceneEntityCfg("target")})
    end_ball_to_target = RewTerm(func=mdp.end_ball_distance_to_target, weight=3, params={"end_ball_contact_sensor_config": SceneEntityCfg("imu_end_ball"), "target_box_config": SceneEntityCfg("target"),"min_val": 0.0, "max_val": 3.0})


@configclass
class EventCfg:
    """ Event specifications"""
    # TODO: Add more during sim2real Transfer stage.

    # reset
    reset_scene = EventTerm(func=mdp.reset_scene_to_default, mode="reset")
    # interval
    push_robot = EventTerm(
        func=mdp.push_by_setting_velocity,
        mode="interval",
        interval_range_s=(10.0, 15.0),
        params={"velocity_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5)}},
    )

@configclass
class TerminationsCfg:
    """ Termination specifications"""

    # time out
    time_out = DoneTerm(func=mdp.time_out, time_out=True)

    # base height, if the contact forces here is larger than 1.0, then terminate
    base_contact = DoneTerm(
        func=mdp.illegal_contact,
        params={"sensor_cfg": SceneEntityCfg("contact_forces_base", body_names="base"), "threshold": 1.0},
    )

@configclass
class CurriculumCfg:
    pass


@configclass
class DMDMEnvCfg(ManagerBasedEnvCfg):

    # Scene settings
    scene: DMDMSceneCfg = DMDMSceneCfg(num_envs=1024, env_spacing=5.3)

    

    # Basic settings:
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()

    # MDP settings:
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()
    curriculum: CurriculumCfg = CurriculumCfg()
    commands = None
    is_finite_horizon: bool = False


    def __post_init__(self):
        """Post initialization."""
        # general settings
        self.sim.device = "cuda:0"
        self.episode_length_s = 20.0
        self.decimation = 2  # env decimation -> 50 Hz control
        # simulation settings
        self.sim.dt = 1/200  # simulation timestep -> 200 Hz physics
        self.sim.gravity = (0.0, 0.0, -9.81)
        #self.sim.physx.solver_type = 0

        # if self.scene.contact_forces is not None:
        #     self.scene.contact_forces.update_period = self.sim.dt
        
        self.sim.render_interval = self.decimation
        self.sim.disable_contact_processing = True


class DMDMEnvCfg_PLAY(ManagerBasedEnvCfg):

    # Scene settings
    scene: DMDMSceneCfg = DMDMSceneCfg(num_envs=10, env_spacing=5.3)

    

    # Basic settings:
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()

    # MDP settings:
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()
    curriculum: CurriculumCfg = CurriculumCfg()
    commands = None


    def __post_init__(self):
        """Post initialization."""
        # general settings
        self.sim.device = "cuda:0"
        self.episode_length_s = 20.0
        self.decimation = 2  # env decimation -> 50 Hz control
        # simulation settings
        self.sim.dt = 1/200  # simulation timestep -> 200 Hz physics
        self.sim.gravity = (0.0, 0.0, -9.81)
        #self.sim.physx.solver_type = 0

        # if self.scene.contact_forces is not None:
        #     self.scene.contact_forces.update_period = self.sim.dt
        
        self.sim.render_interval = self.decimation
        self.sim.disable_contact_processing = True
        self.observations.policy.enable_corruption = False
        self.events.push_robot = None # type: ignore

