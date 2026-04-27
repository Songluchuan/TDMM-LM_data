
# ✨ TDMM-LM: Bridging Facial Understanding and Animation via Language Models
[**🌐 Homepage**](https://songluchuan.github.io/TDMM-LM/) | [**🔬 Paper**](https://arxiv.org/abs/2603.16936) | [**👩‍💻 Code**](https://github.com/Songluchuan/TDMM-LM_data)

## TDMM-LM Dataset
> TDMM-LM Dataset is a large-scale facial animation dataset synthesized with foundation generative models, comprising roughly 80 hours of face-centric video that spans a wide spectrum of emotions,  expressions, and head motions, with each clip paired with its text prompt and 3D facial parameters for training text-driven facial animation/understanding models.

![alt text](assets/TDMMLM.png)


Our dataset enables researchers and practitioners to uncover the strengths, limitations, and potential areas for improvement in text-driven facial animation/understaning models, offering valuable insights into the challenges of generating expressive and emotionally faithful facial behavior.



## 📊 Video Dataset/Annotation [Part-1, \~70hr]


• Videos Download: [**Google drive**](https://drive.google.com/drive/folders/11wWL6vWxxzHJMpSzYlA0uijJkCO10mLC?usp=sharing) (./download_gdrive_folder.sh)

• Language Annotation: As shown in [**json file**](https://github.com/Songluchuan/TDMM-LM_data/blob/main/json/data_part1.json).

## 📊 Video Dataset/Annotation [Part-2, \~10hr]


• Coming Soon.

## 🎵 Audios

• Coming Soon [Synchronized with videos in Part-1].

## 🔧 Tools

• We recommend using [**smirk**](https://github.com/georgeretsi/smirk) or other facial tracking methods to extract the parameters. 

• We provide a [**batch processing script by smirk**](https://github.com/Songluchuan/TDMM-LM_data/tree/main/tools/smirk_inverse) as a reference. 

• We provide a [**batch processing script by spectre**](https://github.com/Songluchuan/TDMM-LM_data/tree/main/tools/spectre_inverse) as a reference. 


## ✏️ Citation
```bibtex
@article{song2026tdmm,
  title={TDMM-LM: Bridging Facial Understanding and Animation via Language Models},
  author={Song, Luchuan and Liu, Pinxin and Liu, Haiyang and Jin, Zhenchao and Tang, Yolo Yunlong and Xu, Zichong and Liang, Susan and Bi, Jing and Corso, Jason J and Xu, Chenliang},
  journal={arXiv preprint arXiv:2603.16936},
  year={2026}
}
```
