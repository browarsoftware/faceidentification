# Face identification with deep neural network

Source code for real time faces identification with USB camera.

Hot to run:
- Run acquire_data.py to gather dataset. Press "s" to save faces.
- Run generate_embedding.py to create embedding.txt file. Inside the file in the second column you may change numbers to names of persons who you want to identify.
- Run run_detection.py, the nearest neighbour classification is used.

Requires:
- Tensorflow with GPU enabled (tested on Tensorflow 2.8)
- keras-vggface (tested on 0.6)
- mtcnn (tested on 0.1.1)

The methodology is based on the papers:

- Omkar M. Parkhi, Andrea Vedaldi and Andrew Zisserman. Deep Face Recognition. In Xianghua Xie, Mark W. Jones, and Gary K. L. Tam, editors, Proceedings of the British Machine Vision Conference (BMVC), pages 41.1-41.12. BMVA Press, September 2015. [pdf](https://www.robots.ox.ac.uk/~vgg/publications/2015/Parkhi15/parkhi15.pdf)
- Qiong Cao, Li Shen, Weidi Xie, Omkar M. Parkhi, Andrew Zisserman, VGGFace2: A dataset for recognising faces across pose and age arXiv:1710.08092 [pdf](https://arxiv.org/abs/1710.08092)

Good tutorials for GPU Tensorflow instalation:

- https://www.youtube.com/watch?v=EmZZsy7Ym-4
- https://www.tensorflow.org/install/source?hl=pl#gpu
- https://www.tensorflow.org/install/pip?hl=pl&_gl=1*4a7na2*_ga*NTE3NTQ5MjA0LjE2Nzk3ODI3OTQ.*_ga_W0YLR4190T*MTY4MjM3OTI2Ni43LjEuMTY4MjM3OTM0NC4wLjAuMA..#windows-native_1
- https://towardsdatascience.com/how-to-finally-install-tensorflow-gpu-on-windows-10-63527910f255
- https://www.tensorflow.org/install/source_windows?hl=pl