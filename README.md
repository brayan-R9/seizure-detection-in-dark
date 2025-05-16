This project aims to develop a deep-learning-based seizure detection system that enhances lowlight video footage, extracts key motion patterns, and classifies seizure events in real time. 
The pipeline includes video enhancement using contrast adjustment techniques, skeletal keypoint extraction for movement tracking, and a deep learning model for accurate classification. Additionally, it provides real-time alerts to caregivers when a seizure is detected.

A key feature of this system is its ability to store and analyze seizure clips for future review, ensuring that critical medical events are documented. By integrating AI-driven motion analysis
with low-light enhancement, this project offers a reliable and efficient solution for real-time patient monitoring, improving response times and patient safety in low-visibility conditions.

The proposed solution introduces a real-time seizure detection system optimized for dark environments using a multi-stage video processing pipeline. The system enhances low-light videos through Gamma Intense Correction (GIC) and Contrast Limited Adaptive Histogram Equalization
(CLAHE) to improve visibility. FastDVDnet is applied to denoise noisy frames while preserving motion integrity.OpenPose extracts keypoints representing human skeletal motion, reducing back ground noise. Small 32×32 pixel patches are extracted around keypoints to focus on critical body
movements. A deep learning model (VSViG) then analyzes these patches and keypoint sequences toclassify seizures based on spatiotemporal motion patterns. If seizure probability exceeds a defined threshold, the system triggers real-time alerts via sound, desktop notifications, and a flashing GUI.
The method ensures non-invasive, real-time, and reliable seizure detection, making it suitable for clinical and home-based monitoring, where traditional EEG-based systems may not be practical.

The Full report of the project has been included in the repository as "Project_Report_Document.pdf"

Dark_video_input :
![Screenshot 2025-05-16 134912](https://github.com/user-attachments/assets/a0672562-ac08-4885-b6fc-00f7ddd8abed)

preprocessed and denoised output :
![Screenshot 2025-05-16 135129](https://github.com/user-attachments/assets/2ef45eb4-054c-4e9e-be76-719a43b6edc3)
This image shows the output after the video frames have been preprocessed and denoised to improve visibility.

keypoint detection :
![Screenshot 2025-05-16 134625](https://github.com/user-attachments/assets/48476de3-bb34-4408-8ea0-7c1c9aafff2d)
Image shows the keypoints being extracted using openpose for pattern analysis to generate seizure probabilities.

final probability generation :
![Screenshot 2025-05-16 134651](https://github.com/user-attachments/assets/574263b9-a98d-4cf0-b173-99e2a6cf9e18)
Final seizure probability generated from the input video using our model after analyzing features and patterns.

Noise comparison :
![Screenshot 2025-05-16 134316](https://github.com/user-attachments/assets/ee073403-0b7a-4f09-bb8f-df964c0be728)
The graph shows the noise level difference between input video and preprocessed video, where we can observe that the noise has been reduced to a good extent in the processed output.

seizure probability graph :
![Screenshot 2025-05-16 134325](https://github.com/user-attachments/assets/f58d4edc-b74d-43a0-a304-8f8c11f2a94a)
Seizure probability generated over the duration of the input video. We can see the probability rises with respect to time signifying that there is a high chance of seizure.



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
