from pathlib import Path
import sys
from collections import OrderedDict
from typing import List, Tuple

ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT))

import numpy as np
import torch
import torch.nn as nn

from .hFT_Transformer.amt import AMT
from .hFT_Transformer.model_spec2midi import (
    Encoder_SPEC2MIDI as Encoder,
    Decoder_SPEC2MIDI as Decoder,
    Model_SPEC2MIDI as BaseSpec2MIDI,
)
from utils import config


DEVICE_DEFAULT = torch.device("cuda" if torch.cuda.is_available() else "cpu")
Array = List | Tuple | np.ndarray | torch.Tensor


class Pipeline(AMT):
    def __init__(
        self,
        path_model: str | None = None,
        device: torch.device = DEVICE_DEFAULT,
        amt: bool = False,
        with_sv: bool = True,
        no_load: bool = False,
        no_model: bool = False,
    ):
        """
        Pipeline for converting audio to MIDI. Contains some methods for
        converting audio to MIDI, models, and configurations.

        Args:
            path_model (str, optional):
                Path to the model. If None, use the default model
                (CONFIG.PATH.AMT or CONFIG.PATH.APC). Defaults to None.
            device (torch.device, optional):
                Device to use for the model. Defaults to auto (CUDA if
                available else CPU).
            amt (bool, optional):
                Whether to use the AMT model. Defaults to False (use the
                APC model).
            with_sv (bool, optional):
                Whether to use the style vector. Defaults to True.
            no_load (bool, optional):
                Do not load the model. Defaults to False.
            no_model (bool, optional):
                Do not own the model. Defaults to False.
        """
        self.device = device
        self.with_sv = with_sv
        if no_model:
            self.model = None
        else:
            self.model = load_model(
                device=self.device,
                amt=amt,
                path_model=path_model,
                with_sv=with_sv,
                no_load=no_load,
            )
        self.config = config.data

    def wav2midi(
        self,
        path_input: str,
        path_output: str,
        sv: None | Array = None,
        silent: bool = True,
    ):
        """
        Convert audio to MIDI.

        Args:
            path_input (str): Path to the input audio file.
            path_output (str): Path to the output MIDI file.
            sv (None | Array, optional): Style vector. Defaults to None.
        """
        if sv is not None:
            sv = torch.tensor(sv)
            if sv.dim() == 1:
                sv = sv.unsqueeze(0)
            if sv.dim() == 2:
                pass
            else:
                raise ValueError(f"Invalid shape of style vector: {sv.shape}")
            sv = sv.to(self.device).to(torch.float32)

        feature = self.wav2feature(path_input)
        infer = config.infer
        if infer.get("stride", False) or infer.get("velocity_fallback", False):
            onset, offset, frame, velocity, velocity_nz = self.transcript_stride_sv(
                feature, sv, stride=infer.get("stride", False)
            )
            if infer.get("velocity_fallback", False):
                velocity = velocity_nz
        else:
            _, _, _, _, onset, offset, frame, velocity = self.transcript(feature, sv, silent)
        if not silent:
            print("Converting to MIDI ...", end=" ", flush=True)
        note = self.mpe2note(
            onset,
            offset,
            frame,
            velocity,
            thred_onset=config.infer.threshold.onset,
            thred_offset=config.infer.threshold.offset,
            thred_mpe=config.infer.threshold.frame,
        )
        note = apply_min_duration(note, config.infer.min_duration, extend=config.infer.get("extend_short", False))
        self.note2midi(note, path_output, 0.0)
        if not silent:
            print("Done.")


    def transcript_stride_sv(self, a_feature, sv=None, stride=True):
        """The B (time-axis) heads with the style vector, plus the velocity argmax over classes 1..127.

        Fork fix A4: `stride=True` is hFT-Transformer's own `transcript_stride` (windows hop by half a
        window, each keeps its centre half, `n_offset = num_frame // 4`), which upstream AMT-APC does
        not call: its `transcript` uses non-overlapping 512-frame windows with 32 frames of context,
        so frames near every 8.2 s join are decoded with almost no context.
        Fork fix A2: `velocity_nz` = argmax over classes 1..127. `mpe2note` drops a note whose onset
        fired but whose velocity argmax (class 0 = "no note") is 0 at that frame.
        """
        cfg = self.config
        num_frame = cfg["input"]["num_frame"]
        n_bins = cfg["feature"]["n_bins"]
        num_note = cfg["midi"]["num_note"]
        margin_b, margin_f = cfg["input"]["margin_b"], cfg["input"]["margin_f"]
        hop, n_offset = (num_frame // 2, num_frame // 4) if stride else (num_frame, 0)
        a_feature = np.array(a_feature, dtype=np.float32)
        rows = int(np.ceil(a_feature.shape[0] / hop) * hop)
        pad_b = np.full([margin_b + n_offset, n_bins], cfg["input"]["min_value"], dtype=np.float32)
        tail = rows - a_feature.shape[0] + (num_frame - n_offset) + margin_f
        pad_f = np.full([tail, n_bins], cfg["input"]["min_value"], dtype=np.float32)
        a_input = torch.from_numpy(np.concatenate([pad_b, a_feature, pad_f], axis=0))
        onset = np.zeros((rows, num_note), dtype=np.float32)
        offset = np.zeros((rows, num_note), dtype=np.float32)
        mpe = np.zeros((rows, num_note), dtype=np.float32)
        velocity = np.zeros((rows, num_note), dtype=np.int16)
        velocity_nz = np.zeros((rows, num_note), dtype=np.int16)
        self.model.eval()
        keep = slice(n_offset, n_offset + hop)
        for i in range(0, a_feature.shape[0], hop):
            input_spec = a_input[i:i + margin_b + num_frame + margin_f].T.unsqueeze(0).to(self.device)
            with torch.no_grad():
                out = self.model(input_spec, sv)
            onset[i:i + hop] = out[5].squeeze(0)[keep].cpu().numpy()
            offset[i:i + hop] = out[6].squeeze(0)[keep].cpu().numpy()
            mpe[i:i + hop] = out[7].squeeze(0)[keep].cpu().numpy()
            vel = out[8].squeeze(0)[keep]
            velocity[i:i + hop] = vel.argmax(2).cpu().numpy()
            velocity_nz[i:i + hop] = (vel[..., 1:].argmax(2) + 1).cpu().numpy()
        return onset, offset, mpe, velocity, velocity_nz


def apply_min_duration(notes, min_duration, extend=False):
    """Upstream's `note2midi` floor drops every note shorter than `min_duration`.

    Fork fix A1 (`extend=True`): keep the note and lengthen it to `min_duration`, never past the
    next onset of the same key. With the offset threshold at 1.0 a fast note's end is where the
    frame head falls, often < 80 ms, and the floor deleted it.
    """
    if not extend:
        return [n for n in notes if n["offset"] - n["onset"] >= min_duration]
    next_onset, out = {}, []
    for n in sorted(notes, key=lambda x: (x["onset"], x["pitch"]), reverse=True):
        m = dict(n)
        if m["offset"] - m["onset"] < min_duration:
            m["offset"] = max(m["offset"], min(m["onset"] + min_duration, next_onset.get(m["pitch"], float("inf"))))
        next_onset[m["pitch"]] = m["onset"]
        out.append(m)
    return sorted(sorted(out, key=lambda x: x["pitch"]), key=lambda x: x["onset"])


class Spec2MIDI(BaseSpec2MIDI):
    def __init__(self, encoder, decoder, sv_dim: int = 0):
        super().__init__(encoder, decoder)
        self.encoder = encoder
        self.decoder = decoder
        delattr(self, "encoder_spec2midi")
        delattr(self, "decoder_spec2midi")
        self.sv_dim = sv_dim
        if sv_dim:
            hidden_size = encoder.hid_dim
            self.fc_sv = nn.Linear(sv_dim, hidden_size)
            self.gate_sv = nn.Sequential(
                nn.Linear(hidden_size, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, hidden_size),
                nn.Sigmoid(),
            )

    def forward(self, x, sv=None):
        h = self.encode(x, sv)
        y = self.decode(h)
        return y

    def encode(self, x, sv=None):
        h = self.encoder(x)
        if self.sv_dim and (sv is not None):
            sv = self.fc_sv(sv)
            _, n_frames, n_bin, _ = h.shape
            sv = sv.unsqueeze(1).unsqueeze(2)
            sv = sv.repeat(1, n_frames, n_bin, 1)
            z = self.gate_sv(h)
            h = z * h + (1 - z) * sv
        return h

    def decode(self, h):
        onset_f, offset_f, mpe_f, velocity_f, attention, \
        onset_t, offset_t, mpe_t, velocity_t = self.decoder(h)
        return (
            onset_f, offset_f, mpe_f, velocity_f, attention,
            onset_t, offset_t, mpe_t, velocity_t
        )


def load_model(
    path_model: str | None = None,
    device: torch.device = DEVICE_DEFAULT,
    amt: bool = False,
    with_sv: bool = True,
    no_load: bool = False,
) -> Spec2MIDI:
    """
    Load the model.

    Args:
        path_model (str, optional):
            Path to the model. If None, use the default model
            (CONFIG.PATH.AMT or CONFIG.PATH.APC). Defaults to None.
        device (torch.device, optional):
            Device to use for the model. Defaults to auto (CUDA if
            available else CPU).
        amt (bool, optional):
            Whether to use the AMT model. Defaults to False (use the
            APC model).
        with_sv (bool, optional):
            Whether to use the style vector. Defaults to True.
        no_load (bool, optional):
            Do not load the model. Defaults to False.
    Returns:
        Spec2MIDI: The model.
    """
    if amt:
        path_model = path_model or str(ROOT / config.path.amt)
    else:
        path_model = path_model or str(ROOT / config.path.apc)

    encoder = Encoder(
        n_margin=config.data.input.margin_b,
        n_frame=config.data.input.num_frame,
        n_bin=config.data.feature.n_bins,
        cnn_channel=config.model.cnn.channel,
        cnn_kernel=config.model.cnn.kernel,
        hid_dim=config.model.transformer.hid_dim,
        n_layers=config.model.transformer.encoder.n_layer,
        n_heads=config.model.transformer.encoder.n_head,
        pf_dim=config.model.transformer.pf_dim,
        dropout=config.model.dropout,
        device=device,
    )
    decoder = Decoder(
        n_frame=config.data.input.num_frame,
        n_bin=config.data.feature.n_bins,
        n_note=config.data.midi.num_note,
        n_velocity=config.data.midi.num_velocity,
        hid_dim=config.model.transformer.hid_dim,
        n_layers=config.model.transformer.decoder.n_layer,
        n_heads=config.model.transformer.decoder.n_head,
        pf_dim=config.model.transformer.pf_dim,
        dropout=config.model.dropout,
        device=device,
    )
    sv_dim = config.model.sv_dim if with_sv else 0
    model = Spec2MIDI(encoder, decoder, sv_dim=sv_dim)
    if not no_load:
        state_dict = torch.load(
            path_model,
            weights_only=True,
            map_location=device
        )
        model.load_state_dict(state_dict, strict=False)
    model.to(device)
    return model


def save_model(model: nn.Module, path: str) -> None:
    """
    Save the model.

    Args:
        model (nn.Module): Model to save.
        path (str): Path to save the model.
    """
    state_dict = model.state_dict()
    correct_state_dict = OrderedDict()
    for key, value in state_dict.items():
        key = key.replace("_orig_mod.", "")
            # If the model has been compiled with `torch.compile()`,
            # "_orig_mod." is appended to the key
        key = key.replace("module.", "")
            # If the model is saved with `torch.nn.DataParallel()`,
            # "module." is appended to the key
        correct_state_dict[key] = value
    torch.save(correct_state_dict, path)
