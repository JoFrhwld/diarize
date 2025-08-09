from dotenv import load_dotenv, dotenv_values
import os
from pathlib import Path
from pyannote.audio import Pipeline
from pyannote.core.annotation import Annotation
from pyannote.audio.pipelines.utils.hook import ProgressHook
import torchaudio
import pympi
import click
import torch
import logging

logger = logging.getLogger("diarize")
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler = logging.StreamHandler()
handler.setFormatter(formatter)
logger.handlers.clear()
logger.addHandler(handler)
logger.setLevel(logging.INFO)

def load_pipeline():
    logger.info("loading pipeline")
    load_dotenv()
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


def do_diarize(path:Path|str, min_speakers:int = 2, max_speakers:int = 4)->Annotation:
  pipeline = load_pipeline()
  logger.info("starting diarization")
  waveform, sample_rate = torchaudio.load(str(path))
  waveform = torchaudio.functional.resample(
    waveform, 
    orig_freq=sample_rate, 
    new_freq=16000
  )

  with ProgressHook() as hook:
    diarization = pipeline(
          {"waveform": waveform, "sample_rate": 16000},
          min_speakers = min_speakers,
          max_speakers = max_speakers,
          hook = hook
      )
  
  return diarization

def write_eaf(diarization:Annotation, out_path:Path|str, audio_file: Path|str) -> None:
  eaf = pympi.Eaf()
  logger.info("writing diarization")
  for turn, _, speaker in diarization.itertracks(yield_label=True):
    if not speaker in eaf.get_tier_names():
      eaf.add_tier(tier_id = speaker)
    eaf.add_annotation(
      id_tier = speaker,
      start = int(turn.start*1000),
      end = int(turn.end*1000),
    )
  eaf.remove_tier(id_tier = "default")
  if audio_file:
    eaf.add_linked_file(file_path=str(audio_file))
  
  eaf.to_file(out_path)

@click.command()
@click.argument(
   "path",
   type = click.Path(path_type=Path)
)
def main(path:Path|str):
    logging.basicConfig(level=logging.INFO)
    diarization = do_diarize(path)
    write_eaf(
       diarization=diarization,
       out_path=Path(path).with_suffix(".eaf"),
       audio_file=path
    )
    logger.info("job finished")


if __name__ == "__main__":
    main()
