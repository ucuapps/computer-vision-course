# Computer Vision, Syllabus Fall 2019

## General Information

**Teachers:**
* [Oles Dobosevych](https://apps.ucu.edu.ua/en/personal/oles-dobosevych), email: dobosevych@ucu.edu.ua
* [Igor Krashenyi](https://scholar.google.com.ua/citations?user=J3GqVSMAAAAJ&hl), email: igor.krashenyi@gmail.com
* [Tetiana Martyniuk](https://apps.ucu.edu.ua/en/personal/tmartyniuk), email: t.martynyuk@ucu.edu.ua
* [Yaroslava Lochman](https://apps.ucu.edu.ua/en/personal/ylochman), email: lochman@ucu.edu.ua

**[CMS](https://cms.ucu.edu.ua/course/view.php?id=2248)**

**Course Description:** Computer Vision has become ubiquitous in our society, with applications in search, image understanding, apps, mapping, medicine, drones, and self-driving cars. Core to many of these applications are image classification, localization, detection and other visual tasks. Recent developments in neural network approaches have greatly advanced the performance of these state-of-the-art visual recognition systems. This course will give students an understanding and practical experience with many important deep CNN models applied for specific tasks of classification, segmentation, detection, recognition and restoration.

**Goals:** The main goal of the course is to discover the most important problems in modern Computer Vision, and approach them with powerful CNN architectures. Students will be able to implement, train, validate and debug their own neural networks and gain a detailed understanding of cutting-edge research in computer vision.


## Course Format
* Lectures + Seminars
* Every Tuesday (Sep 17, 2019 –– Dec 24, 2019)
* 8:30 — 11:30 @ Sheptytsky Center, Room 308

## Schedule and Syllabus
|   	                           	      |Сlass Hours    |Self work hours    |Course Materials   |
|---	                                  |---	          |---	              |---	     	         |
|**Module 1. Introduction <br> (Sep 17; Oct 1)**       |9              |18                 |[Deep Learning for Computer Vision](https://github.com/lyubonko/ucu2019)<br>[\[Assignment #1\]]() |
|Intro to CNN    	                      |   	          |   	              |[\[slides\]](./module1-intro/lecture1_intro.pdf)   	                          |
|CNNs for Image Classification          |   	          |   	              |[\[slides\]](./module1-intro/lecture2_CNN.pdf)   	                          |
|CNNs for OCR    	                      |   	          |   	              |   	                          |
|**Module 2. Image Segmentation <br> (Oct, 11—14)**                    |9              |18                   | |
|Problem Statement<br>CNNs for Semantic Segmentation|            	          |   	              |[\[slides\]](./module2-segmentation/lecture1_image_segmentation.pdf)   	                          |
|Retina Blood Vessel Segmentation         |   	          |   	              |[\[code & data\]](https://drive.google.com/open?id=1eVKexzcwlbF-FEDg1unjSmRk0TgRZbBM)   	                          |
|Deep Automatic Portrait Matting      |   	          |   	                  |[\[Assignment #2\]](https://drive.google.com/open?id=1VV7bzGQH717wxK6kSBtR96R5KvHajFpO)                               |
|**Module 3. Generative Models <br> (Nov, 22—25)**  |9              |18                   |[\[Assignment #3\]]() |
|Autoencoders                           |   	          |   	              |   	                          |
|VAE, GANs                              |   	          |   	              |   	                          |
|CNNs for Image Restoration             |   	          |   	              |   	                          |
|**Module 4. Instance-level Recognition <br> (Dec, 12—14)**|9       |18                 |[\[Assignment #4\]]() |
|Modules and Objectives for Instance-level Recognition| |   	              |[\[slides\]]() <br> papers: [FPN](https://arxiv.org/pdf/1612.03144.pdf), [Focal Loss](https://arxiv.org/pdf/1708.02002.pdf)   	                          |
|R-CNN family, YOLO, SSD for Object Detection|   	      |   	              |[\[slides\]]() <br> papers:  <br>[R-CNN](https://arxiv.org/pdf/1311.2524.pdf), [Fast R-CNN](https://arxiv.org/pdf/1504.08083.pdf), [Faster R-CNN](https://arxiv.org/pdf/1506.01497.pdf),  <br>[YOLO](https://arxiv.org/pdf/1506.02640.pdf), [YOLO9000](https://arxiv.org/pdf/1612.08242.pdf), [YOLOv3](https://arxiv.org/pdf/1804.02767.pdf),  <br>[SSD](https://arxiv.org/pdf/1512.02325.pdf)   	                          |
|Mask R-CNN for Instance Segmentation   |   	          |   	              |[\[slides\]]() <br> papers: [Mask R-CNN](https://arxiv.org/pdf/1703.06870.pdf)   	                          |
|**\*optional: Face Recognition**       |               |                   |papers:  <br> [DeepFace](https://www.cv-foundation.org/openaccess/content_cvpr_2014/papers/Taigman_DeepFace_Closing_the_2014_CVPR_paper.pdf), [FaceNet](https://www.cv-foundation.org/openaccess/content_cvpr_2015/papers/Schroff_FaceNet_A_Unified_2015_CVPR_paper.pdf), [Deep Face Recognition](http://cis.csuohio.edu/~sschung/CIS660/DeepFaceRecognition_parkhi15.pdf)                               |
|**Total**                              |36             |72                 |                               |


## Course Policy
* No plagiarism and other violation of academic integrity is allowed. Be sure to obey the [academic code of honour of UCU](https://s3-eu-central-1.amazonaws.com/ucu.edu.ua/wp-content/uploads/2017/04/Polozhennya_pro_plagiat.pdf).
* A completed assignment should be submitted to `cms` as an archive (named as **Name-Surname-CV-HW#**) or a link to the repository.

## Grading Policy
* Assignment #1: 25%
* Assignment #2: 25%
* Assignment #3: 25%
* Assignment #4: 25%


## Recommended Materials
**main:**
* cs231n ([course notes](http://cs231n.github.io) | [video lectures](https://www.youtube.com/playlist?list=PL3FW7Lu3i5JvHM8ljYj-zLfQRF3EO8sYv))
* Multiple View Geometry in Computer Vision by Hartley and Zisserman. ([book](https://github.com/pranjals16/cs676/blob/master/Hartley%2C%20Zisserman%20-%20Multiple%20View%20Geometry%20in%20Computer%20Vision.pdf))
* Elements of Geometry in Computer Vision by Pajdla ([book](https://cw.felk.cvut.cz/courses/GVG/2013/Lecture/GVG-2013-Lecture.pdf))

**optional:**
* Computer vision: algorithms and applications by Szeliski. ([book](http://szeliski.org/Book/drafts/SzeliskiBook_20100903_draft.pdf))
* Computer vision: a modern approach by Forsyth and Ponce. ([book](http://cmuems.com/excap/readings/forsyth-ponce-computer-vision-a-modern-approach.pdf))
* cs131 ([course notes](https://github.com/StanfordVL/CS131_notes))
* EPFL Deep Learning ([course](https://fleuret.org/ee559))
* CS294-158 Deep Unsupervised Learning ([course](https://sites.google.com/view/berkeley-cs294-158-sp19/home))
