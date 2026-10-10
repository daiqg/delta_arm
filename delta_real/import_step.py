"""Inspect a STEP assembly with OpenCascade; preserve names and world placements.

CAD intake only: geometric volume is not mass, assembly pose is not joint zero.
Install cadquery-ocp in a separate CAD environment (see CAD_REVIEW.md).
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from pathlib import Path

from OCP.BRepAdaptor import BRepAdaptor_Surface
from OCP.BRepBndLib import BRepBndLib
from OCP.BRepGProp import BRepGProp
from OCP.Bnd import Bnd_Box
from OCP.GProp import GProp_GProps
from OCP.GeomAbs import GeomAbs_Cylinder, GeomAbs_Sphere
from OCP.IFSelect import IFSelect_RetDone
from OCP.STEPCAFControl import STEPCAFControl_Reader
from OCP.TCollection import TCollection_ExtendedString
from OCP.TDataStd import TDataStd_Name
from OCP.TDF import TDF_Label
from OCP.collections import Sequence_TDF_Label as TDF_LabelSequence
from OCP.TDocStd import TDocStd_Document
from OCP.TopAbs import TopAbs_FACE, TopAbs_SOLID
from OCP.TopExp import TopExp_Explorer
from OCP.TopLoc import TopLoc_Location
from OCP.TopoDS import TopoDS
from OCP.XCAFDoc import XCAFDoc_DocumentTool


def name_of(label):
    attr = TDataStd_Name()
    return attr.Get().ToExtString() if label.FindAttribute(TDataStd_Name.GetID_s(), attr) else ''


def xyz(point):
    return [point.X(), point.Y(), point.Z()]


def geometry(shape, assembly=False):
    if assembly:
        return dict(bounds_mm=None, size_mm=None, geometric_volume_mm3=None,
                    volume_centroid_mm=None, solid_count=0, analytic_surfaces=[])
    box = Bnd_Box()
    BRepBndLib.AddOptimal_s(shape, box, False, False)
    bounds = xyz(box.CornerMin()) + xyz(box.CornerMax())
    props = GProp_GProps()
    BRepGProp.VolumeProperties_s(shape, props)
    solids = TopExp_Explorer(shape, TopAbs_SOLID)
    count = 0
    while solids.More():
        count += 1
        solids.Next()
    features = []
    faces = TopExp_Explorer(shape, TopAbs_FACE)
    face_index = 0
    while faces.More():
        surface = BRepAdaptor_Surface(TopoDS.Face(faces.Current()), True)
        feature = None
        if surface.GetType() == GeomAbs_Cylinder:
            cylinder = surface.Cylinder()
            feature = dict(type='cylinder', radius_mm=cylinder.Radius(),
                           axis_point_mm=xyz(cylinder.Location()), axis=xyz(cylinder.Axis().Direction()))
        elif surface.GetType() == GeomAbs_Sphere:
            sphere = surface.Sphere()
            feature = dict(type='sphere', radius_mm=sphere.Radius(), center_mm=xyz(sphere.Location()))
        if feature:
            feature['face_index'] = face_index
            features.append(feature)
        face_index += 1
        faces.Next()
    return dict(bounds_mm=bounds, size_mm=[bounds[i+3]-bounds[i] for i in range(3)],
                geometric_volume_mm3=props.Mass(), volume_centroid_mm=xyz(props.CentreOfMass()),
                solid_count=count, analytic_surfaces=features)


def load_step(path):
    doc = TDocStd_Document(TCollection_ExtendedString('delta_cad'))
    reader = STEPCAFControl_Reader()
    reader.SetNameMode(True)
    reader.SetColorMode(True)
    if reader.ReadFile(str(path.resolve())) != IFSelect_RetDone or not reader.Transfer(doc):
        raise RuntimeError('STEP import failed')
    shape_tool = XCAFDoc_DocumentTool.ShapeTool_s(doc.Main())
    roots = TDF_LabelSequence()
    shape_tool.GetFreeShapes(roots)
    return doc, shape_tool, roots


def step_metadata(path):
    """Keep source material claims separate from trusted geometry."""
    text = path.read_text(encoding='utf-8')
    records = dict(re.findall(r'#(\d+)=(.*?);', text, re.S))
    decode = lambda s: re.sub(r'\\X2\\(.*?)\\X0\\',
                             lambda m: bytes.fromhex(m[1]).decode('utf-16-be'), s, flags=re.S)
    materials = []
    for value in records.values():
        if value.startswith("DESCRIPTIVE_REPRESENTATION_ITEM('"):
            materials.append(decode(value))
    return dict(schema='STEP AP214', source_length_unit='mm',
                density_values_raw=sorted(set(re.findall(
                    r"'density measure',\s*POSITIVE_RATIO_MEASURE\((.*?)\)", text))),
                material_descriptions=sorted(set(materials)),
                material_status='Unverified CAD attributes; do not infer physical mass.')


def render_preview(rows, shapes, output):
    """Render real B-rep triangulations, without substituting primitive geometry."""
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    from OCP.BRep import BRep_Tool
    from OCP.BRepMesh import BRepMesh_IncrementalMesh
    triangles, colors = [], []
    for row, shape in zip(rows, shapes):
        if row['assembly']:
            continue
        path = row['path']
        color = '#4b607a' if 'DM-J' in path else '#a3afb8'
        if '主动臂' in path: color = '#ee9b43'
        if '从动臂' in path: color = '#3f9fa8'
        if '末端平台' in path: color = '#c95852'
        BRepMesh_IncrementalMesh(shape, 0.8, False, 0.3, True)
        explorer = TopExp_Explorer(shape, TopAbs_FACE)
        while explorer.More():
            face = TopoDS.Face(explorer.Current())
            loc = TopLoc_Location()
            mesh = BRep_Tool.Triangulation_s(face, loc)
            if mesh is not None:
                points = np.array([xyz(mesh.Node(i).Transformed(loc.Transformation()))
                                   for i in range(1,mesh.NbNodes()+1)])
                indices = np.array([mesh.Triangle(i).Get() for i in range(1,mesh.NbTriangles()+1)]) - 1
                triangles.extend(points[indices])
                colors.extend([color]*len(indices))
            explorer.Next()
    triangles = np.asarray(triangles)
    lo, hi = triangles.min(axis=(0,1)), triangles.max(axis=(0,1))
    center, span = (lo+hi)/2, max(hi-lo)
    fig = plt.figure(figsize=(14,7))
    for index, angles in enumerate([(22,-55),(90,-90)]):
        ax = fig.add_subplot(1,2,index+1,projection='3d')
        ax.add_collection3d(Poly3DCollection(triangles, facecolors=colors, edgecolors='none', linewidths=0))
        ax.set(xlim=(center[0]-span/2,center[0]+span/2),
               ylim=(center[1]-span/2,center[1]+span/2), zlim=(center[2]-span/2,center[2]+span/2),
               xlabel='CAD X (mm)', ylabel='CAD Y (mm)', zlabel='CAD Z (mm)')
        ax.set_box_aspect((1,1,1)); ax.view_init(*angles)
    fig.suptitle('Received STEP assembly | orange: upper arms | teal: lower rods | red: platform')
    fig.tight_layout(); fig.savefig(output/'assembly_preview.png', dpi=160); plt.close(fig)


def inspect(path, output, preview=False):
    doc, tool, roots = load_step(path)
    rows, shapes = [], []

    def visit(label, parent_location, parent_path):
        location = parent_location.Multiplied(tool.GetLocation_s(label))
        definition = TDF_Label()
        if not tool.GetReferredShape_s(label, definition):
            definition = label
        name = name_of(label) or name_of(definition)
        instance_path = parent_path + [name]
        # Definition shapes are local; apply accumulated occurrence placement once.
        shape = tool.GetShape_s(definition).Located(location)
        children = TDF_LabelSequence()
        tool.GetComponents_s(definition, children)
        row = dict(index=len(rows), path='/'.join(instance_path), name=name,
                   definition=name_of(definition), depth=len(parent_path), assembly=children.Length()>0,
                   transform=[[location.Transformation().Value(i,j) for j in range(1,5)] for i in range(1,4)])
        row.update(geometry(shape, row['assembly']))
        rows.append(row)
        shapes.append(shape)
        if len(parent_path) <= 1:
            print(f'Imported {row["index"]}: {name}', flush=True)
        for i in range(1, children.Length()+1):
            visit(children.Value(i), location, instance_path)
        if row['assembly']:
            leaves = [r for r in rows[row['index']+1:] if not r['assembly']]
            row['bounds_mm'] = [min(r['bounds_mm'][i] for r in leaves) for i in range(3)] + [
                max(r['bounds_mm'][i] for r in leaves) for i in range(3,6)]
            row['size_mm'] = [row['bounds_mm'][i+3]-row['bounds_mm'][i] for i in range(3)]
            row['solid_count'] = sum(r['solid_count'] for r in leaves)

    for i in range(1, roots.Length()+1):
        visit(roots.Value(i), TopLoc_Location(), [])
    output.mkdir(parents=True, exist_ok=True)
    report = dict(source_filename=path.name, source_bytes=path.stat().st_size,
                  source_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                  coordinate_units='mm (OpenCascade STEP reader target)',
                  caveat='Static CAD geometry only. Volume/centroid are not measured mass/COM.',
                  source_metadata=step_metadata(path), occurrences=rows)
    (output/'assembly.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    with (output/'parts.csv').open('w', encoding='utf-8-sig', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['index','path','definition','assembly','solids','size_x_mm','size_y_mm','size_z_mm','volume_mm3'])
        for row in rows:
            writer.writerow([row['index'],row['path'],row['definition'],row['assembly'],row['solid_count'],
                             *row['size_mm'],row['geometric_volume_mm3']])
    print(f'Occurrences: {len(rows)}; leaves: {sum(not r["assembly"] for r in rows)}')
    if preview:
        render_preview(rows, shapes, output)
    return rows, shapes


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('step', type=Path)
    parser.add_argument('--output', type=Path, default=Path(__file__).resolve().parent/'cad')
    parser.add_argument('--preview', action='store_true', help='Render CAD geometry to a PNG (requires matplotlib).')
    args = parser.parse_args()
    inspect(args.step, args.output, args.preview)
