import warnings
from dotenv import load_dotenv
import os
from pathlib import Path

from polars import date
with warnings.catch_warnings():
  warnings.simplefilter("ignore")
  from pyannote.audio import Pipeline
  from pyannote.core.annotation import Annotation
  from pyannote.audio.pipelines.utils.hook import ProgressHook
import librosa
import pympi
import click
import torch
import logging
from diarize.logging import make_loggers, make_file_handler, err_log
from diarize.interval_status import isolated_speech, short_silence, silence_silence, speech_small_pause, speech_speech
from aligned_textgrid import AlignedTextGrid, TierGroup, SequenceTier, SequenceInterval, custom_classes
import numpy as np
import time
import datetime



logger = make_loggers("diarize")
#logging.basicConfig(level = logging.INFO)

@err_log(logger)
def load_pipeline():
    logger.info("loading pipeline")
    success = load_dotenv()
    logger.info(f".env loaded: {success}")
    with warnings.catch_warnings():
      warnings.simplefilter("ignore")
      pipeline = Pipeline.from_pretrained(
          "pyannote/speaker-diarization-3.1",
          use_auth_token=os.getenv("HUGGINGFACE_ACCESS_TOKEN")
      )
    if torch.cuda.is_available():
       device = torch.device("cuda")
       logger.info("running on cuda")
    else:
       device = torch.device("cpu")
       logger.info("running on cpu")
    pipeline.to(device)
    return pipeline

@err_log(logger)
def do_diarize(path:Path|str, min_speakers:int = 2, max_speakers:int = 4)->Annotation:
  pipeline = load_pipeline()
  logger.info("Loading audio")
  waveform, sample_rate = librosa.load(str(path), sr = 16000)
  waveform_t = torch.tensor(waveform.reshape(1, -1))

  logger.info(f"Audio info - samples: {waveform.size}; sr: {sample_rate}; dur: {str(datetime.timedelta(seconds = round(waveform.size/sample_rate)))}")

  logger.info("Beginning Diarization")
  start = time.time()
  with ProgressHook() as hook:
    with warnings.catch_warnings():
      warnings.simplefilter("ignore")
      diarization = pipeline(
            {"waveform": waveform_t, "sample_rate": 16000},
            min_speakers = min_speakers,
            max_speakers = max_speakers,
            hook = hook
        )
  end = time.time()
  logger.info(f"Diarization finished ({str(datetime.timedelta(seconds=round(end-start)))})")

  return diarization

@err_log(logger)
def make_atg(diarization:Annotation)->AlignedTextGrid:
  logger.info("Creating TextGrid")
  tiers = dict()
  Diar, = custom_classes(["Diar"])
  for turn, id, speaker in diarization.itertracks(yield_label=True):
    logger.debug(f"{turn.start}, {turn.end}, {id}, {speaker}")
    if (turn.end - turn.start) < 0.2:
      continue
    if not speaker in tiers:
      logger.debug(f"Adding tier for {speaker}")
      stier = SequenceTier(entry_class=Diar)
      stier.name = speaker
      tiers[speaker] = stier

    d = Diar((turn.start, turn.end, id))
    tiers[speaker].append(d)
  logger.debug("Creating tiergroups")
  tgs = [
    TierGroup([tier])
    for tier in tiers.values()
  ]
  logger.debug("Creating aligned textgrid")
  atg = AlignedTextGrid(tgs)
  
  max_end = np.array([
    tg[0].ends.max()
    for tg in atg
  ]).max()

  for tg in atg:
    tier = tg[0]
    tier.cleanup()
    first = tier.first
    last = tier.last
    if first.start > 0:
      logger.debug("adding front pad")
      front_pad = Diar((0, first.start, ""))
      tier.append(front_pad)
    if last.end < max_end:
      logger.debug("adding end pad")
      end_pad = Diar((last.end, max_end, ""))
      tier.append(end_pad)
  
  atg.cleanup()

  return atg

@err_log(logger)
def coallece(atg: AlignedTextGrid):
  logger.debug("Coalecing intervals")
  for tg in atg:
    tier = tg[0]
    interval = tier.first
    while not interval is tier.last:
      if interval.label == "#":
        break
      logger.debug(f"{interval.id}")
      if speech_small_pause(interval):
        logger.debug("speech small pause")
        interval.fuse_rightward()
      if speech_speech(interval):
        logger.debug("speech speech")
        interval.fuse_rightward()
      if short_silence(interval):
        logger.debug("short silence")
        interval.fuse_rightward()
      if silence_silence(interval):
        logger.debug("silence silence")
        interval.fuse_rightward()

      iso_speech, side = isolated_speech(interval)
      if iso_speech and side in ["left", "right"]:
        if side == "left":
          logger.debug("Iso speech, fusing left")
          interval.fuse_leftward()
          if speech_speech(interval.prev):
            interval.fuse_leftward()
        if side == "right":
          logger.debug("Iso speech, fusing right")          
          interval.fuse_rightward()
          if speech_speech(interval):
            interval.fuse_rightward()

      elif iso_speech:
        logger.debug(f"Popping interval of duration {interval.duration:.03}")
        try:
          interval.fuse_leftward()
        except Exception as e:
          logger.error("Leftward error")
          logger.error(e.with_traceback)
        try:
          interval.fuse_rightward()
        except Exception as e:
          logger.error("Rightward error")
          logger.error(e.with_traceback)
        interval.label = ""
        continue

      if interval is tier.last:
        break
      
      interval = interval.fol

  return atg


@err_log(logger)
def write_eaf(atg:AlignedTextGrid, out_path:Path|str, audio_file: Path|str) -> None:
  eaf = pympi.Eaf()
  logger.info("Writing diarization")
  n_annotations = -np.array([
    len(tg[0])
    for tg in atg.tier_groups
  ])
  for idx in n_annotations.argsort():
    tg = atg.tier_groups[int(idx)]
    tier = tg[0]
    eaf.add_tier(tier_id=f"Speaker-{idx}")
    for interval in tier:
      if len(str(interval.label)) > 0:
        eaf.add_annotation(
          id_tier = f"Speaker-{idx}",
          start = int(interval.start * 1000),
          end = int(interval.end * 1000),
          value = interval.label
        )
  eaf.remove_tier(id_tier = "default")
  if audio_file:
    eaf.add_linked_file(file_path=str(audio_file))
  
  eaf.to_file(out_path)
  logger.info(f"Diarization written to {str(out_path)}")

@click.command()
@click.argument(
   "path",
   type = click.Path(path_type=Path)
)
@click.option(
  "--min_speakers",
  default = 2,
  type = click.IntRange(1,10)
)
@click.option(
  "--max_speakers",
  default = 4,
  type = click.IntRange(2,10)
)
@click.option(
  "--debug",
  is_flag = True
)
def main(path:Path|str, min_speakers: int, max_speakers: int, debug: bool):
    start = time.time()
    fhandler = make_file_handler(path)
    logger.addHandler(fhandler)
    if debug:
      logger.setLevel(logging.DEBUG)
    else:
      logger.setLevel(logging.INFO)
    logger.info(f"Running diarize with min_speakers={min_speakers}, max_speakers={max_speakers}")
    diarization = do_diarize(path, min_speakers=min_speakers, max_speakers=max_speakers)
    atg = make_atg(diarization)
    atg = coallece(atg)
    write_eaf(
       atg=atg,
       out_path=Path(path).with_suffix(".eaf"),
       audio_file=path
    )
    logger.info("Job Finished")
    logger.removeHandler(fhandler)
    end = time.time()
    logger.info(f"Total time: {str(datetime.timedelta(seconds=round(end-start)))}")


if __name__ == "__main__":
    main()
