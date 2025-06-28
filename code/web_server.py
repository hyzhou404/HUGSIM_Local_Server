import sys
import os
import pickle
import json
import threading
import enum
import hugsim_env
from collections import deque, OrderedDict
from datetime import datetime
from typing import Any, List
from dataclasses import dataclass
sys.path.append(os.getcwd())

from moviepy import ImageSequenceClip
from fastapi import FastAPI, Body
from fastapi.responses import HTMLResponse, Response
from omegaconf import OmegaConf, DictConfig
import open3d as o3d
import numpy as np
import gymnasium
import uvicorn

from sim.utils.sim_utils import traj2control, traj_transform_to_global
from sim.utils.score_calculator import hugsim_evaluate
from glob import glob
from random import shuffle
from time import time

def to_video(observations: List[Any], output_path: str):
    frames = []
    for obs in observations:
        row1 = np.concatenate([obs['CAM_FRONT_LEFT'], obs['CAM_FRONT'], obs['CAM_FRONT_RIGHT']], axis=1)
        row2 = np.concatenate([obs['CAM_BACK_RIGHT'], obs['CAM_BACK'], obs['CAM_BACK_LEFT']], axis=1)
        frame = np.concatenate([row1, row2], axis=0)
        frames.append(frame)
    clip = ImageSequenceClip(frames, fps=4)
    clip.write_videofile(output_path)


class FifoDict:
    def __init__(self, max_size: int):
        self.max_size = max_size
        self._order_dict = OrderedDict()
        self.locker = threading.Lock()
    
    def push(self, key: str, value: Any):
        with self.locker:
            if key in self._order_dict:
                self._order_dict.move_to_end(key)
                return
            if len(self._order_dict) >= self.max_size:
                self._order_dict.popitem(last=False)
            self._order_dict[key] = value
    
    def get(self, key: str) -> Any:
        return self._order_dict.get(key, None)


@dataclass
class SceneConfig:
    name: str
    cfg: DictConfig


@dataclass
class EnvExecuteResult:
    cur_scene_done: bool
    done: bool


def _get_scene_list(base_output: str) -> List[SceneConfig]:
    """
    Load the scene configurations from the YAML files.
    Returns:
        List[SceneConfig]: A list of scene configurations.
    """
    
    scene_list = []
    # for data_type in ['kitti360', 'waymo', 'nuscenes', 'pandaset']:
    for data_type in ['kitti360', "nuscenes"]:
        base_path = os.path.join(os.path.dirname(__file__), "web_server_config", f'{data_type}_base.yaml')
        camera_path = os.path.join(os.path.dirname(__file__), "web_server_config", f'{data_type}_camera.yaml')
        kinematic_path = os.path.join(os.path.dirname(__file__), "web_server_config", 'kinematic.yaml')

        base_config = OmegaConf.load(base_path)
        camera_config = OmegaConf.load(camera_path)
        kinematic_config = OmegaConf.load(kinematic_path)
        
        scenarios_list = glob(f"/app/app_datas/ss/scenarios/{data_type}/*.yaml")
        # if data_type == 'waymo':
        #     scenarios_list = [
        #         # f"/app/app_datas/ss/scenarios/{data_type}/scene-113792265837-extreme-01.yaml",
        #         f"/app/app_datas/ss/scenarios/{data_type}/scene-322492347634-extreme-01.yaml",
        #         f"/app/app_datas/ss/scenarios/{data_type}/scene-131421903137-easy-00.yaml",
        #         # f"/app/app_datas/ss/scenarios/{data_type}/scene-131421903137-medium-02.yaml",
        #     ]
        # if data_type == 'pandaset':
        #     scenarios_list = [
        #         f"/app/app_datas/ss/scenarios/{data_type}/scene-034-easy-00.yaml",
        #         # f"/app/app_datas/ss/scenarios/{data_type}/scene-131421903137-medium-02.yaml",
        #         # f"/app/app_datas/ss/scenarios/{data_type}/scene-322492347634-extreme-01.yaml"
        #     ]

        for scenario_path in scenarios_list:
            scenario_config = OmegaConf.load(scenario_path)
            cfg = OmegaConf.merge(
                {"scenario": scenario_config},
                {"base": base_config},
                {"camera": camera_config},
                {"kinematic": kinematic_config}
            )

            model_path = os.path.join(cfg.base.model_base, cfg.scenario.scene_name)
            model_config = OmegaConf.load(os.path.join(model_path, 'cfg.yaml'))
            model_config.update({"model_path": f"/app/app_datas/ss/scenes/{data_type}/{cfg.scenario.scene_name}"})
            cfg.update(model_config)
            cfg.base.output_dir = base_output
            
            scene_list.append(SceneConfig(name=cfg.scenario.scene_name, cfg=cfg))
    return scene_list


class EnvHandler:
    """A class to handle the environment for HUGSim.
    This can include multiple scene and configurations.
    """
    def __init__(self, scene_list: List[SceneConfig], base_output: str):
        self._lock = threading.Lock()
        self.scene_list = scene_list
        self.base_output = base_output
        self.env = None
        self.reset_env()

    def _switch_scene(self, scene_index: int):
        """
        Switch to a different scene based on the index.
        Args:
            scene_index (int): The index of the scene to switch to.
        """
        if scene_index < 0 or scene_index >= len(self.scene_list):
            raise ValueError("Invalid scene index.")
        
        self.close()
        self.cur_scene_index = scene_index
        scene_config = self.scene_list[scene_index]
        self._log(f"Switch to scene: {scene_config.name}_{scene_config.cfg.scenario.mode}")
        print(f"Switch to scene: {scene_config.name}_{scene_config.cfg.scenario.mode}")
        
        self.cur_otuput = os.path.join(self.base_output, 
                                       f"{scene_config.name}_{scene_config.cfg.scenario.mode}")
        os.makedirs(self.cur_otuput, exist_ok=True)
        self.env = gymnasium.make('hugsim_env/HUGSim-v0', cfg=scene_config.cfg, output=self.cur_otuput)
        self._scene_cnt = 0
        self._scene_done = False
        self._save_data = {'type': 'closeloop', 'frames': []}
        self._observations_save = []
        self._obs, self._info = self.env.reset()


    def close(self):
        """
        Close the environment and release resources.
        """
        if self.env is not None:
            del self.env
        self.env = None
        self._log("Environment closed.")

    def reset_env(self):
        """
        Reset the environment and initialize variables.
        """
        self._log_list = deque(maxlen=100)
        self._done = False
        self._score_list = []
        self._switch_scene(0)
        self._log("Environment reset complete.")
    
    def get_current_state(self):
        """
        Get the current state of the environment.
        """
        return {
            "obs": self._obs,
            "info": self._info,
        }

    @property
    def has_done(self) -> bool:
        """
        Check if the episode is done.
        Returns:
            bool: True if the episode is done, False otherwise.
        """
        return self._done
    
    @property
    def has_scene_done(self) -> bool:
        """
        Check if the current scene is done.
        Returns:
            bool: True if the current scene is done, False otherwise.
        """
        return self._scene_done

    @property
    def log_list(self) -> deque:
        """
        Get the log list.
        Returns:
            deque: The log list containing recent log messages.
        """
        return self._log_list

    def execute_action(self, plan_traj: np.ndarray) -> EnvExecuteResult:
        """
        Execute the action based on the planned trajectory.
        Args:
            plan_traj (Any): The planned trajectory to follow.
        Returns:
            bool: True if the episode is done, False otherwise.
        """
        acc, steer_rate = traj2control(plan_traj, self._info)
        action = {'acc': acc, 'steer_rate': steer_rate}
        self._log("Executing action:", action)
    
        self._obs, _, terminated, truncated, self._info = self.env.step(action)
        self._scene_cnt += 1
        self._scene_done = terminated or truncated or self._scene_cnt > 400

        imu_plan_traj = plan_traj[:, [1, 0]]
        imu_plan_traj[:, 1] *= -1
        global_traj = traj_transform_to_global(imu_plan_traj, self._info['ego_box'])
        self._save_data['frames'].append({
            'time_stamp': self._info['timestamp'],
            'is_key_frame': True,
            'ego_box': self._info['ego_box'],
            'obj_boxes': self._info['obj_boxes'],
            'obj_names': ['car' for _ in self._info['obj_boxes']],
            'planned_traj': {
                'traj': global_traj,
                'timestep': 0.5
            },
            'collision': self._info['collision'],
            'rc': self._info['rc']
        })
        self._observations_save.append(self._obs['rgb'])
        
        if not self._scene_done:
            return EnvExecuteResult(cur_scene_done=False, done=False)

        with open(os.path.join(self.cur_otuput, 'data.pkl'), 'wb') as wf:
            pickle.dump([self._save_data], wf)
        
        ground_xyz = np.asarray(o3d.io.read_point_cloud(os.path.join(self.cur_otuput, 'ground.ply')).points)
        scene_xyz = np.asarray(o3d.io.read_point_cloud(os.path.join(self.cur_otuput, 'scene.ply')).points)
        results = hugsim_evaluate([self._save_data], ground_xyz, scene_xyz)
        with open(os.path.join(self.cur_otuput, 'eval.json'), 'w') as f:
            json.dump(results, f)
        self._score_list.append(results.copy())
        to_video(self._observations_save, os.path.join(self.cur_otuput, 'video.mp4'))
        
        self._log(f"Scene {self.cur_scene_index} completed. Evaluation results saved.")
    
        if self.cur_scene_index < len(self.scene_list) - 1:
            self._switch_scene(self.cur_scene_index + 1)
            return EnvExecuteResult(cur_scene_done=True, done=False)

        self._done = True
        return EnvExecuteResult(cur_scene_done=True, done=True)

    def _log(self, *messages):
        log_message = f"[{str(datetime.now())}]" + " ".join([str(msg) for msg in messages]) + "\n"
        with self._lock:
            self._log_list.append(log_message)


def get_env_handler():
    base_output = "/app/app_datas/env_output"
    scene_list = _get_scene_list(base_output)
    output = os.path.join(base_output, "hugsim_env")
    os.makedirs(output, exist_ok=True)
    return EnvHandler(scene_list, base_output=output)


app = FastAPI()

_result_dict= FifoDict(max_size=100)
env_handler = get_env_handler()


def _load_numpy_ndarray_json_str(json_str: str) -> np.ndarray:
    """
    Load a numpy ndarray from a JSON string.
    """
    data = json.loads(json_str)
    return np.array(data["data"], dtype=data["dtype"]).reshape(data["shape"])


@app.post("/reset")
async def reset_endpoint():
    """
    Reset the environment.
    """
    env_handler.reset_env()
    return {"success": True}


@app.get("/get_current_state")
async def get_current_state_endpoint():
    """
    Get the current state of the environment.
    """
    state = env_handler.get_current_state()
    data = {
        "done": env_handler.has_done,
        "cur_scene_done": env_handler.has_scene_done,
        "state": state,
    }
    return Response(content=pickle.dumps(data), media_type="application/octet-stream")


@app.post("/execute_action")
async def execute_action_endpoint(
    plan_traj: str = Body(..., embed=True),
    transaction_id: str = Body(..., embed=True),
):
    """
    Execute the action based on the planned trajectory.
    Args:
        plan_traj (str): The planned trajectory in JSON format.
        transaction_id (str): The unique transaction ID for caching results.
        env_handler (EnvHandler): The environment handler instance.
    Returns:
        Response: The response containing the execution result.
    """
    cache_result = _result_dict.get(transaction_id)
    if cache_result is not None:
        return Response(content=cache_result, media_type="application/octet-stream")

    if env_handler.has_done:
        result = pickle.dumps({"done": True, "cur_scene_done": True, "state": env_handler.get_current_state()})
        _result_dict.push(transaction_id, result)
        return Response(content=result, media_type="application/octet-stream")

    plan_traj = _load_numpy_ndarray_json_str(plan_traj)
    execute_result = env_handler.execute_action(plan_traj)
    if execute_result.done:
        result = pickle.dumps({"done": execute_result.done, "cur_scene_done": execute_result.cur_scene_done, "state": env_handler.get_current_state()})
        _result_dict.push(transaction_id, result)
        return Response(content=result, media_type="application/octet-stream")
    
    state = env_handler.get_current_state()
    result = pickle.dumps({"done": execute_result.done, "cur_scene_done": execute_result.cur_scene_done, "state": state})
    _result_dict.push(transaction_id, result)
    return Response(content=result, media_type="application/octet-stream")


@app.get("/submition_info")
async def main_page_endpoint():
    """
    Endpoint to display the submission logs.
    """
    log_str = "\n".join(env_handler.log_list)
    html_content = f"""
        <html><body><pre>{log_str}</pre></body></html>
        <script>
            setTimeout(function() {{
                window.location.reload();
            }}, 5000);
        </script>
    """
    return HTMLResponse(content=html_content)


uvicorn.run(app, host="0.0.0.0", port=7860, workers=1)
