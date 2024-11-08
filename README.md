# Handle-counter
  App that counts pieces on frame in real time for galvanic line using Tensorflow object detection

  This is my portfolio project, so it likely won't have much use other than to show it in resume.
and it designed specificaly for needs of factory i worked on. the main condition of success was high precision no less than 99%
  
  Work on project includes:
  - gathering and labeling dataset
  - choosing detection model and tuning config
  - training model on local GPU with CUDA and CUDNN
  - developing app that uses video stream to count pieces
  - implementing app into work process on factory
  All stages of development made by me, from gathering and labeling data to training model, writing app and implementing it on the work station
  ![image](https://github.com/haghehog-hee/Handle-counter/assets/110155576/4c4ef8b7-9539-460d-b9b3-37b665f05407)

the project contains three scripts: 
________________________________________________________________________________________________________________________
  main.py is an app, which takes a direct video stream from camera and, on button click, performs object detection, 
visualising results and writing number of found objects in text format. It uses model efficientdetd0, which i trained on local GPU.
Due to the fact, that all images in dataset are frames from one camera, it will perform poorly in any other instance
  
  EfficientdetD0 can only process images of resolution 512x512 and camera provides much higher resolution. So in order to not lose that information, 
app splits input image into 3 parts (that is because there are most commonly 3 frames on the line) and processes them separately, this technique greatly increases performance.
  
  By technical reasons, camera located slightly above frame and not perpendiculary to it, so I implemented Affine transformations in order to make shot more orthogonal,
that also improved accuracy of detection
________________________________________________________________________________________________________________________
  autoannotations.py is a tool, that i used to reduce labor intencity of data annotation process, and also to monitor model performance

Currently there are 120 types of pieces added to train set, and labeling them all manually would be very time consuming.
this script runs existing model on images that need to be labeled and writes results into annotation files, which require little supervising after,
but speeds up data annotation process approximately 2-3 times.
  Manual labeling was made in labelimg utilite
  
  Huge part was preventing overfit, as training dataset still stays relatively small where some types contain only few hundred examples.
  So model's pipeline includes many options for data augmentation, to bypass this problem
________________________________________________________________________________________________________________________

AN ULTIMATE DATASET GENERATOR.py is another dataset tool

It takes random background from "backgrounds" folder and randomly puts objects over it, automatically generating annotation file for that picture. 
This way it can create large amounts of examples for dataset imitating real scenario
