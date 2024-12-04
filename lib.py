import numpy as np

note_to_idx = {
    'c': 0,
    'd': 2,
    'e': 4,
    'f': 5,
    'g': 7,
    'a': 9,
    'b': 11
}

#returns index of a note from 0-11
def parse_note(note):
    idx = note_to_idx.get(note[0])

    #accidentals
    acc = len(note)
    if acc > 1:
        if note[1] == '#':
            idx += (acc-1)
        elif note[1] == 'b':
            idx -= (acc-1)
        else:
            print('unknown accidental')
    return idx

#returns np array of a chord text representation of the notes
def chord_to_np(chord):
    out = np.zeros(12)
    for note in chord.split():
        idx = parse_note(note)
        out[idx] = 1
    return out


print(parse_note('f##'))
print(chord_to_np('c e g'))