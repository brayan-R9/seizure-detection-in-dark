This project aims to develop a deep-learning-based seizure detection system that enhances lowlight video footage, extracts key motion patterns, and classifies seizure events in real time. 
The pipeline includes video enhancement using contrast adjustment techniques, skeletal keypoint extraction for movement tracking, and a deep learning model for accurate classification. Additionally, it provides real-time alerts to caregivers when a seizure is detected.

A key feature of this system is its ability to store and analyze seizure clips for future review, ensuring that critical medical events are documented. By integrating AI-driven motion analysis
with low-light enhancement, this project offers a reliable and efficient solution for real-time patient monitoring, improving response times and patient safety in low-visibility conditions.

The proposed solution introduces a real-time seizure detection system optimized for dark environments using a multi-stage video processing pipeline. The system enhances low-light videos through Gamma Intense Correction (GIC) and Contrast Limited Adaptive Histogram Equalization
(CLAHE) to improve visibility. FastDVDnet is applied to denoise noisy frames while preserving motion integrity.OpenPose extracts keypoints representing human skeletal motion, reducing back ground noise. Small 32×32 pixel patches are extracted around keypoints to focus on critical body
movements. A deep learning model (VSViG) then analyzes these patches and keypoint sequences toclassify seizures based on spatiotemporal motion patterns. If seizure probability exceeds a defined threshold, the system triggers real-time alerts via sound, desktop notifications, and a flashing GUI.
The method ensures non-invasive, real-time, and reliable seizure detection, making it suitable for clinical and home-based monitoring, where traditional EEG-based systems may not be practical.


Citation :

@inproceedings{xu2024vsvig,
  title={VSViG: Real-Time Video-Based Seizure Detection via Skeleton-Based Spatiotemporal ViG},
  author={Xu, Yankun and Wang, Junzhe and Chen, Yun-Hsuan and Yang, Jie and Ming, Wenjie and Wang, Shuang and Sawan, Mohamad},
  booktitle={European Conference on Computer Vision},
  pages={228--245},
  year={2024},
  organization={Springer}
}

@inproceedings{Xu2023VSViG,
  title={VSViG: Real-time Video-based Seizure Detection via Skeleton-based Spatiotemporal ViG},
  author={Yankun Xu and Junzhe Wang and Yun-Hsuan Chen and Jie Yang and Wenjie Ming and Shuangquan Wang and Mohamad Sawan},
  booktitle={arXiv preprint arXiv:2311.14775},
  year={2023}
}
