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

## ⚙️ Quick Start
### 1. Environment
```bash
conda create -n misp python=3.9 pytorch torchaudio cudatoolkit=11.8 -c pytorch -y
conda activate misp
pip install -r requirements.txt
```

### 2. Speech Enhancement (GSS)
We use the **guided source separation (GSS)** implementation from  
<https://github.com/desh2608/gss>.

```bash
git clone https://github.com/desh2608/gss external/gss
# follow the GSS README to enhance 8‑ch recordings, e.g.
python external/gss/apply_gss.py    --audio_dir /path/to/MISP/far_audio    --rt60 0.3 --mic_format misp    --out_dir /path/to/MISP/far_audio_gss
```

### 3. Speech Recognition (AVSR & Fine‑tuning)
For recognition we reuse the **AVSR** recipe developed in our previous work  
<https://github.com/mispchallenge/MISP-ICME-AVSR>.

```bash
git clone https://github.com/mispchallenge/MISP-ICME-AVSR external/avsr
# see external/avsr/README.md for end‑to‑end training / inference
```

### 4. Summarisation Utilities
```bash
# generate brief & detailed summaries with DeepSeek LLM
python summary_by_deepseek.py
```
Alternative LLM back‑ends are available: `gemini`, `kimi`, `ollama`, `qwen`.
---


## 📄 Citation
```
@inproceedings{chen2025misp,
  title = "{MISP - Meeting}: A Real-World Dataset with Multimodal Cues for Long-form Meeting Transcription and Summarization",
  author = "Chen, Hang and Yang, Chao-Han Huck and Gu, Jia-Chen and Siniscalchi, Sabato Marco and Du, Jun",
  booktitle = "Proceedings of the 63st Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)",
  year = "2025",
  publisher = "Association for Computational Linguistics",
  pages = "1--14"}
```

## 🙏 Acknowledgements
MISP‑Meeting is a joint effort of USTC, NVIDIA Research, UCLA, and the University of Palermo.  
We thank the 60+ research groups who provided early feedback and the volunteers who annotated and validated the corpus.
