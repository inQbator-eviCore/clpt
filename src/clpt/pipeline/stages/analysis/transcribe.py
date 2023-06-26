# coding=utf-8
"""NLP DocumentCleaner stage for cleaning the CLAO.

DocumentCleaner includes removing stop words, converting to lower case, and excluding punctuations.
"""

from espnet2.bin.asr_inference import Speech2Text
from espnet_model_zoo.downloader import ModelDownloader
from overrides import overrides

from src.clao.audio_clao import Audio, AudioCLAO
from src.clao.text_clao import Text
from src.clpt.pipeline.stages.pipeline_stage import PipelineStage
from src.constants.annotation_constants import RAW_AUDIO, RAW_TEXT

lang = 'en'
fs = 16000
tag = 'Shinji Watanabe/spgispeech_asr_train_asr_conformer6_n_fft512_hop_length256_raw_en_unnorm_bpe5000_valid.acc.ave'
d = ModelDownloader()

# It may takes a while to download and build models
speech2text = Speech2Text(
    **d.download_and_unpack(tag),
    device="cpu",
    minlenratio=0.0,
    maxlenratio=0.0,
    ctc_weight=0.3,
    beam_size=10,
    batch_size=1,
    nbest=1
)

STOPWORD = '<stopword>'


class Transcribe(PipelineStage):
    """Convert audio to it's corresponding text"""

    @overrides
    def __init__(self, **kwargs):
        super(Transcribe, self).__init__(**kwargs)

    def process(self, clao_info: AudioCLAO) -> None:
        """Clean the raw texts in CLAO(s) and add the cleaned text to CLAO(s).

        Args:
            clao_info (TextCLAO): The CLAO information to process
        """
        audio_obj = clao_info.get_annotations(Audio, {'description': RAW_AUDIO})
        transcribed_text, *_ = speech2text(audio_obj.raw_data)[0]
        clao_info.insert_annotation(Text, Text(transcribed_text, RAW_TEXT))
