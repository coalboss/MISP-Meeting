# MISP-Meeting: Multimodal Dataset for Long-Form Meeting Transcription and Summarization

**MISP-Meeting** is a real‑world Mandarin meeting corpus that combines **speech, panoramic video, and text** to enable research on automatic meeting transcription and summarization (AMTS) and other multimodal perception tasks.

```
📅  120+ hours   🎙️ 8‑ch far‑field audio   🎧 per‑speaker head‑set audio   🎥 360° video
📝 sentence‑level transcripts   📄 human‑refined brief & detailed summaries
```

> Accepted at **ACL 2025** (Resource Track).  
> If you use MISP‑Meeting in your work, please cite our paper (see below).
>
> ## ✨ Key Features
| Category | Details |
|----------|---------|
| **Scale** | 125.15 h raw audio‑visual recordings; 163 real meetings; 274 speakers; 23 rooms |
| **Modalities** | 8‑channel circular microphone array, per‑speaker near‑field mic, 360° RGB video |
| **Noise Conditions** | Real meeting rooms with typing, door slams, fan noise, cross‑talk |
| **Metadata** | Room geometry, speaker demographics (age, profession), topic labels |
| **Human Labels** | Sentence boundaries (≤ ±100 ms), manual transcripts (> 99 % acc.), 2‑pass expert summaries |
| **Licensing** | CC BY‑NC‑ND 4.0 (research‑only, free upon authorisation) |

## 🔗 Download
1. **Sign the licence** on the [official dataset page](https://challenge.xfyun.cn/misp_dataset).
2. Files are hosted on an OSS mirror supporting `aria2c`/`wget`.
3. Verify integrity with the provided checksums.

> Problems? Open an [issue](../../issues) or e‑mail the maintainers.


## 📄 Citation
```
@inproceedings{chen2025mispmeeting,
  title     = {MISP‑Meeting: A Real‑World Dataset with Multimodal Cues for Long‑form Meeting Transcription and Summarization},
  author    = {Hang Chen and Jun Du and Sabato Marco Siniscalchi and et al.},
  booktitle = {Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (ACL)},
  year      = {2025}
}
```

## 🙏 Acknowledgements
MISP‑Meeting is a joint effort of USTC, NVIDIA Research, UCLA, and the University of Palermo.  
We thank the 60+ research groups who provided early feedback and the volunteers who annotated and validated the corpus.
