from pathlib import Path

import mido
import numpy as np


def parse_midi(path: Path) -> np.ndarray:
  """open midi file and return np.array of (onset, offset, note, velocity) rows"""
  midi = mido.MidiFile(path)

  time = 0
  sustain = False
  events = []
  for message in midi:
    time += message.time

    if message.type == 'control_change' and message.control == 64 and (message.value >= 64) != sustain:
      # sustain pedal state has just changed
      sustain = message.value >= 64
      event_type = 'sustain_on' if sustain else 'sustain_off'
      event = {'index': len(events), 'time': time, 'type': event_type, 'note': None, 'velocity': 0}
      events.append(event)

    if 'note' in message.type:
      # MIDI offsets can be either 'note_off' events or 'note_on' with zero velocity
      velocity = message.velocity if message.type == 'note_on' else 0
      event = dict(index=len(events), time=time, type='note', note=message.note, velocity=velocity, sustain=sustain)
      events.append(event)

  notes = []
  for i, onset in enumerate(events):
    if onset['velocity'] == 0:
      continue

    # find the next note_off message
    offset = next(n for n in events[i + 1:] if n['note'] == onset['note'] or n is events[-1])

    if offset['sustain'] and offset is not events[-1]:
      # if the sustain pedal is active at offset, find when the sustain ends
      offset = next(n for n in events[offset['index'] + 1:] if n['type'] == 'sustain_off' or n['note'] == onset['note'] or n is events[-1])

    note = onset['time'], offset['time'], onset['note'], onset['velocity'] / 128.0
    notes.append(note)

  return np.array(notes)
