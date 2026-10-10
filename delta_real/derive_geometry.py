"""Derive Delta linkage measurements from the received v9 assembly features.

Selections are explicit and fail when the source assembly changes. These are
CAD measurements, not a calibrated hardware configuration or joint limits.
"""
import argparse
import json
from pathlib import Path

import numpy as np


def derive(report):
    rows = report['occurrences']

    def named(name):
        found = [r for r in rows if r['name'] == name and r['depth'] == 1]
        if len(found) != 1:
            raise ValueError(f'Expected one top-level occurrence: {name}')
        return found[0]

    def child_spheres(parent):
        children = [r for r in rows if r['path'].startswith(parent['path'] + '/') and not r['assembly']]
        if len(children) != 6:
            raise ValueError('Expected six ball joints per end')
        spheres = []
        for row in children:
            faces = [f for f in row['analytic_surfaces'] if f['type'] == 'sphere']
            if len(faces) != 1:
                raise ValueError(f'Expected one spherical surface: {row["path"]}')
            spheres.append((row, faces[0]))
        return spheres

    upper = child_spheres(named('球关节轴承(上):1'))
    lower = child_spheres(named('球关节轴承（下）:1'))
    up = np.array([f['center_mm'] for _, f in upper])
    low = np.array([f['center_mm'] for _, f in lower])
    # Source occurrence order pairs each branch and side. Independently check
    # the six actual carbon tube axes below before accepting this correspondence.
    elbow = up.reshape(3,2,3).mean(axis=1)
    shoulder, axes, arm_rows = [], [], []
    for i in range(3):
        row = named(f'主动臂:{i+1}')
        face = [f for f in row['analytic_surfaces'] if f['type']=='cylinder' and abs(f['radius_mm']-32.5)<1e-5]
        if len(face) != 1:
            raise ValueError('Upper-arm shoulder selection no longer matches source')
        axis, point = np.array(face[0]['axis']), np.array(face[0]['axis_point_mm'])
        shoulder.append(point + axis*np.dot(elbow[i]-point,axis))
        axes.append(axis); arm_rows.append(row)
    shoulder = np.array(shoulder)
    origin = shoulder.mean(axis=0)
    platform = low.mean(axis=0)
    z_axis = np.cross(shoulder[1]-shoulder[0],shoulder[2]-shoulder[0])
    z_axis /= np.linalg.norm(z_axis)
    if np.dot(z_axis,platform-origin)>0:
        z_axis *= -1
    # First shaft is transverse Y; X is perpendicular to that shaft in the base plane.
    y_axis = -np.array(axes[0]); y_axis -= np.dot(y_axis,z_axis)*z_axis
    y_axis /= np.linalg.norm(y_axis)
    x_axis = np.cross(y_axis,z_axis)
    rotation = np.array([x_axis,y_axis,z_axis])
    transform = np.eye(4); transform[:3,:3]=rotation; transform[:3,3]=-rotation@origin/1000
    to_base = lambda p: ((np.asarray(p)-origin)@rotation.T/1000).tolist()

    tube_names = ['5mm内径碳杆:1','5mm内径碳杆(镜像):1',
                  '5mm内径碳杆:2','5mm内径碳杆(镜像):2',
                  '5mm内径碳杆:3','5mm内径碳杆(镜像):3']
    rod_checks = []
    for i,name in enumerate(tube_names):
        tube = next(r for r in rows if r['name']==name and '/从动臂:1/' in r['path'])
        face = next(f for f in tube['analytic_surfaces'] if f['type']=='cylinder' and abs(f['radius_mm']-5)<1e-5)
        point, axis = np.array(face['axis_point_mm']), np.array(face['axis'])
        distances = [float(np.linalg.norm(np.cross(p-point,axis))) for p in [up[i],low[i]]]
        direction=(low[i]-up[i])/np.linalg.norm(low[i]-up[i])
        angle=float(np.degrees(np.arccos(np.clip(abs(np.dot(direction,axis)),0,1))))
        # A loose correspondence guard must not hide a smaller source CAD mismatch.
        if max(distances)>1.0 or angle>0.1:
            raise ValueError(f'Ball pair is not aligned with carbon tube: {name}: {distances}')
        rod_checks.append(dict(tube_occurrence=tube['index'], endpoint_axis_distances_mm=distances,
                               axis_angle_deg=angle, cad_offset_requires_review=max(distances)>0.1))
    branches=[]
    for i in range(3):
        arm_vector = (elbow[i]-shoulder[i])@rotation.T
        axis = np.array(axes[i])@rotation.T
        # Choose the joint axis so increasing angle lifts the upper arm.
        if np.dot(np.cross(axis,arm_vector),[0,0,1]) < 0: axis *= -1
        branches.append(dict(cad_branch=i+1, source_upper_arm_occurrence=arm_rows[i]['index'],
                             source_upper_ball_occurrences=[r['index'] for r,_ in upper[2*i:2*i+2]],
                             source_lower_ball_occurrences=[r['index'] for r,_ in lower[2*i:2*i+2]],
                             shoulder_center_m=to_base(shoulder[i]), joint_axis=axis.tolist(),
                             upper_ball_centers_m=to_base(up[2*i:2*i+2]),
                             lower_ball_centers_m=to_base(low[2*i:2*i+2]),
                             platform_anchors_m=((low[2*i:2*i+2]-platform)@rotation.T/1000).tolist(),
                             upper_length_m=float(np.linalg.norm(arm_vector)/1000),
                             reference_angle_rad=float(np.arctan2(arm_vector[2],np.linalg.norm(arm_vector[:2]))),
                             upper_pair_spacing_m=float(np.linalg.norm(up[2*i]-up[2*i+1])/1000),
                             lower_pair_spacing_m=float(np.linalg.norm(low[2*i]-low[2*i+1])/1000),
                             lower_lengths_m=(np.linalg.norm(low[2*i:2*i+2]-up[2*i:2*i+2],axis=1)/1000).tolist()))
    return dict(status='cad_measured_not_hardware_calibrated', source_sha256=report['source_sha256'],
                base_frame_definition='Origin: mean shoulder centers in upper ball midplanes; +X toward branch 1; +Z away from platform; right handed.',
                cad_origin_mm=origin.tolist(), rotation_base_from_cad=rotation.tolist(),
                transform_base_from_cad_m=transform.tolist(),
                transform_rule='Convert CAD coordinates mm to m first, then apply homogeneous transform.',
                platform_reference='Mean of six lower spherical-surface centers; not automatically tool mounting face.',
                platform_reference_position_m=to_base(platform),
                source_lengths_unit='mm', stored_lengths_unit='m', branches=branches,
                rod_axis_checks=rod_checks,
                motor_to_branch_mapping=None, encoder_zero_rad=None, joint_limits_rad=None,
                mass_inertia_status='STEP material assignments are not accepted; measurements needed')


def plot_geometry(result, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig=plt.figure(figsize=(12,6))
    for index in range(2):
        ax=fig.add_subplot(1,2,index+1,projection='3d' if index==0 else None)
        dims=3 if index==0 else 2
        for b in result['branches']:
            s=np.array(b['shoulder_center_m'])*1000
            u=np.array(b['upper_ball_centers_m'])*1000
            l=np.array(b['lower_ball_centers_m'])*1000
            e=u.mean(axis=0)
            ax.plot(*np.array([s,e]).T[:dims],color='darkorange',lw=4)
            for side in range(2): ax.plot(*np.array([u[side],l[side]]).T[:dims],color='teal',lw=2)
            ax.plot(*u.T[:dims],color='grey'); ax.plot(*l.T[:dims],color='grey')
            ax.scatter(*s[:dims],color='black'); ax.text(*s[:dims],f'  Branch {b["cad_branch"]}')
        ax.set(xlabel='Base X (mm)',ylabel='Base Y (mm)',xlim=(-320,320),ylim=(-320,320))
        if index==0:
            ax.set(zlabel='Base Z (mm)',zlim=(-480,80)); ax.set_box_aspect((640,640,560)); ax.view_init(22,-55)
        else:
            ax.set_aspect('equal');ax.grid(True);ax.set_title('Top view')
    fig.suptitle('CAD linkage centers | upper ~200 mm | lower ~393.988 mm | pair spacing 67 mm')
    fig.tight_layout();fig.savefig(output,dpi=160);plt.close(fig)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--assembly',type=Path,default=Path(__file__).resolve().parent/'cad/assembly.json')
    parser.add_argument('--plot',action='store_true')
    args=parser.parse_args()
    result=derive(json.loads(args.assembly.read_text(encoding='utf-8')))
    out=args.assembly.parent/'geometry.json'
    out.write_text(json.dumps(result,indent=2,ensure_ascii=False),encoding='utf-8')
    if args.plot:plot_geometry(result,args.assembly.parent/'linkage_dimensions.png')
    maximum=max(max(c['endpoint_axis_distances_mm']) for c in result['rod_axis_checks'])
    print(f'Saved {out}; maximum ball-center/carbon-tube-axis offset: {maximum:.6f} mm (see review flags).')
