from aligned_textgrid import SequenceInterval
import numpy as np

def speech_small_pause(
    interval:SequenceInterval,
    short: float = 0.1
)->bool:
  if interval.fol.label == "#":
    return False
  
  speech = len(interval.label) > 0
  fol_pause = len(interval.fol.label) < 1
  fol_short = interval.fol.duration < short
  return speech and fol_pause and fol_short

def speech_speech(
    interval: SequenceInterval
) -> bool:
  if interval.fol.label == "#":
    return False
  
  speech = len(interval.label) > 0
  fol_speech = len(interval.fol.label) > 0

  return speech and fol_speech


def short_silence(
    interval: SequenceInterval,
    short: float = 0.1
) -> bool:
  if interval.fol.label == "#":
    return False  
  
  silence = len(interval.label) < 1
  is_short = interval.duration < short

  return silence and is_short

def silence_silence(
    interval: SequenceInterval
)->bool:

  if interval.fol.label == "#":
    return False
  
  silence = len(interval.label) < 1
  fol_silence = len(interval.fol.label) < 1

  return silence and fol_silence

def isolated_speech(
    interval: SequenceInterval,
    short: float = 0.2
) -> (bool, str):
  if interval.fol.label == '#':
    return (False, "none")

  speech = len(interval.label) > 0
  is_short = interval.duration < short

  is_short_speech = speech and is_short
  if not is_short_speech:
    return (False, "none")
  
  pre_silence = len(interval.prev.label) < 1
  pre_dur = interval.prev.duration
  fol_silence = len(interval.fol.label) < 1
  fol_dur = interval.fol.duration

  if not pre_silence and not fol_silence:
    return (True, "none")
  
  if pre_silence and not fol_silence and pre_dur < 0.3:
    return (True, "left")

  if not pre_silence and fol_silence and fol_dur < 0.3:
    return (True, "right")
  
  sides = np.array([pre_dur, fol_dur])

  which_min = sides.argmin()
  the_min = sides.min()

  if the_min < 0.3:
    return (True, ["left", "right"][which_min])
  
  return (True, "none")