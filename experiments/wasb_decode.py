"""Balanced FFmpeg selection avoids expression parser depth limits."""


def selection_expression(frames):
    terms=[f'eq(n\\,{int(frame)})' for frame in frames]
    if not terms:raise ValueError('Empty frame selection')
    while len(terms)>1:
        terms=[f'({terms[i]})+({terms[i+1]})' if i+1<len(terms) else terms[i] for i in range(0,len(terms),2)]
    return terms[0]
