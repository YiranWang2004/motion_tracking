#!/usr/bin/env python3
"""Rigidly rotate a dual CFGen bundle about world Z; local quantities stay fixed."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np


def rotate_bundle(source, output, degrees):
    source, output = Path(source).resolve(), Path(output).resolve()
    if output.exists():
        raise FileExistsError(output)
    with np.load(source, allow_pickle=False) as data:
        arrays = {key: data[key].copy() for key in data.files}
    angle = np.deg2rad(degrees)
    c, s = np.cos(angle), np.sin(angle)
    rotation = np.array([[c,-s,0],[s,c,0],[0,0,1]])
    def vector(value):
        return (np.asarray(value) @ rotation.T).astype(np.asarray(value).dtype)
    def quaternion(value):
        q = np.asarray(value)
        w,x,y,z = np.moveaxis(q, -1, 0)
        a,b = np.cos(angle/2), np.sin(angle/2)
        return np.stack([a*w-b*z,a*x-b*y,a*y+b*x,a*z+b*w],axis=-1).astype(q.dtype)
    changed=[]
    for key,value in arrays.items():
        legacy = key.startswith(('robot_0_', 'robot_1_'))
        if (key.startswith('training_') and key.endswith(('_pos_w','_lin_vel_w','_ang_vel_w'))) or (legacy and key.endswith('_pos') and value.shape[-1]==3):
            arrays[key]=vector(value);changed.append(key)
        elif (key.startswith('training_') and key.endswith('_quat_w')) or (legacy and key.endswith('_quat')):
            arrays[key]=quaternion(value);changed.append(key)
    metadata=json.loads(str(arrays['metadata_json'].item()))
    if 'box_initial_pose' in metadata:
        pose=np.asarray(metadata['box_initial_pose'],dtype=float)
        metadata['box_initial_pose']=np.r_[vector(pose[:3]),quaternion(pose[3:])].tolist()
    if 'box_target_position' in metadata:
        metadata['box_target_position']=vector(np.asarray(metadata['box_target_position'],dtype=float)).tolist()
    if 'box_target_yaw' in metadata:
        metadata['box_target_yaw']=float(metadata['box_target_yaw']+angle)
    metadata['world_rotation']={'yaw_degrees':degrees,'origin_w':[0,0,0],
        'source':str(source),'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest()}
    arrays['metadata_json']=np.asarray(json.dumps(metadata,separators=(',',':')))
    output.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(output,**arrays)
    return changed


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',required=True)
    parser.add_argument('--output',required=True)
    parser.add_argument('--yaw-degrees',type=float,default=-90)
    args=parser.parse_args()
    if not np.isfinite(args.yaw_degrees):
        parser.error('yaw must be finite')
    print('Rotated world fields:',rotate_bundle(args.source,args.output,args.yaw_degrees))
