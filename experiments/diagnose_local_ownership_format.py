"""Distinguish actual abstentions from incomplete local model responses."""
import argparse
import collections
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]


def format_summary(records):
    rows = []
    for answer in records:
        text = str(answer.get('response', '')).strip().lower()
        rows.append(dict(exact_unclear=bool(re.fullmatch(r'unclear[.!]?', text)),
            exact_label=bool(re.fullmatch(r'(home|away|stopped|unclear)[.!]?', text)),
            empty=not bool(text), done_reason=answer.get('done_reason'),
            eval_count=answer.get('eval_count'), thinking_nonempty=bool(answer.get('thinking', '')),
            allowed_word_count=len(set(re.findall(r'\b(home|away|stopped|unclear)\b', text)))))
    return dict(samples=len(rows), exact_unclear=sum(x['exact_unclear'] for x in rows),
        empty=sum(x['empty'] for x in rows),
        syntax_rejections=sum(not x['empty'] and not x['exact_label'] for x in rows),
        done_reasons=dict(collections.Counter(x['done_reason'] for x in rows)),
        eval_counts=dict(collections.Counter(x['eval_count'] for x in rows)),
        thinking_nonempty=sum(x['thinking_nonempty'] for x in rows),
        allowed_word_counts=dict(collections.Counter(x['allowed_word_count'] for x in rows)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--instruct', action='store_true')
    parser.add_argument('--montage', action='store_true')
    args = parser.parse_args()
    if args.instruct and args.montage:
        parser.error('Choose one inference stage')
    root = ROOT/'data/control_local_ownership_v1'
    folder = root/('inference_montage' if args.montage else 'inference_instruct' if args.instruct else 'inference')
    records = [json.loads(p.read_text()) for p in sorted(folder.glob('answer_*.json'))]
    result = format_summary(records)
    name = 'format_diagnostics_montage.json' if args.montage else 'format_diagnostics_instruct.json' if args.instruct else 'format_diagnostics_replay.json'
    (root/name).write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
