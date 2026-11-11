"""First target join: complete uniform public-camera localization accounting."""
import json
import xml.etree.ElementTree as ET
import numpy as np
from player_representation import ROOT,digest,write_json
from prepare_wasb_external import validate

FOLDER=ROOT/'data/control_wasb_v1'


def labels(path,n):
    tree=ET.parse(path);result={};seen=[]
    for track in tree.getroot().findall('track'):
        for point in track.findall('points'):
            frame=int(point.attrib['frame']);seen.append(frame)
            values=[p.text for p in point.findall('attribute') if p.attrib.get('name')=='used_in_game']
            if len(values)!=1 or values[0] not in ('0','1'):raise ValueError('Unknown public in-game flag')
            if point.attrib['outside']=='0' and values[0]=='1':
                if frame in result:raise ValueError('Two in-game balls')
                x,y=map(float,point.attrib['points'].split(','));result[frame]=(x,y,point.attrib['occluded']=='0')
    if not seen or min(seen)<0 or max(seen)>=n:raise ValueError('XML zero-index/video range mismatch')
    return result,dict(min_frame=min(seen),max_frame=max(seen),used_entries=len(result),missing_frame_rule='Invisible, matching official soccer loader')


def main():
    plan=validate();source=FOLDER/'external_inputs';output=FOLDER/'external_assessment';output.mkdir(parents=True,exist_ok=False);records=[]
    source_summary=json.loads((source/'summary.json').read_text());predictions=FOLDER/'external_outputs';manifest=json.loads((predictions/'manifest.json').read_text())
    for name,sha in manifest['cache_sha256'].items():
        if digest(predictions/name)!=sha:raise ValueError('Immutable external detections changed')
    for camera in source_summary['records']:
        key=camera['camera'];truth,coverage=labels(source/(key+'.xml'),int(camera['video']['nb_read_frames']));rows=json.loads((predictions/(key+'_predictions.json')).read_text())
        for mode in ('temporal','frame'):
            selected=[r for r in rows if r['mode']==mode];tp=fp=fn=tn=wrong=visible=0
            for row in selected:
                x,y,v=truth.get(row['frame'],(-1.,-1.,False));visible+=v
                if v and row['visible']:
                    if np.hypot(row['x']-x,row['y']-y)<plan['distance_original_pixels']:tp+=1
                    else:fp+=1;fn+=1;wrong+=1
                elif v:fn+=1
                elif row['visible']:fp+=1
                else:tn+=1
            precision=tp/(tp+fp) if tp+fp else 0.;recall=tp/(tp+fn) if tp+fn else 0.;f1=2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0.
            passed=len(selected)==plan['centers_per_camera'] and visible>=32 and f1>=.8 and recall>=.75
            records.append(dict(camera=key,mode=mode,centers=len(selected),visible=visible,invisible=len(selected)-visible,tp=tp,fp=fp,fn=fn,tn=tn,wrong_location=wrong,precision=precision,recall=recall,f1=f1,passed=passed,annotation_coverage=coverage))
    passed=all(r['passed'] for r in records if r['mode']=='temporal') and manifest['peak_GiB']<=2
    write_json(output/'selection.json',dict(passed=passed,records=records,interpretation='Same-play cross-camera checkpoint/adapter sanity, not unseen-match/native-possession evidence',wrong_location_counted_as_fp_and_fn=True,source_publisher_legacy_recall_omits_wrong_location_so_not_used=True))
    write_json(output/'manifest.json',dict(plan_sha256=digest(ROOT/'experiments/WASB_EXTERNAL_PLAN.json'),cache_sha256={p.name:digest(p) for p in sorted(output.iterdir()) if p.is_file()}));print((output/'selection.json').read_text(),flush=True)


if __name__=='__main__':main()
