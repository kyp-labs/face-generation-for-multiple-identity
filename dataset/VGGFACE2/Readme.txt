In this VGGFace2 Dataset, we provide loosely cropped faces for each identity. 

After downloading, unzip the training and test files,

-- In the train or test folder, each identity is named as 'n<classID>' with 6 digits padding with zeros, e.g.'n000001'

-- The cropped face images are stored under identity folder, and each face image will have unique name, e.g. 'n000001/0019_01.jpg',
where '0019_01.jpg' is named in a way "<imageID>_<faceID>.jpg". This is because it is possible to have one image with several faces.

-- The whole dataset is split to training (8631 identities) and test (500 identities) sets. The identities in the two sets are disjoint. 

-- The 'meta' folder includes the following files:

   -- train.txt: the training image list, each row has the unique image name, e.g. 'n000002/0001_01.jpg'

   -- test.txt: the testing image list, each row has the unique image name, e.g. 'n000001/0019_01.jpg'

   -- identity_meta.csv: the identity information with 5 columns, i.e., Class_ID, Name, Sample_Num, Flag, 
      (1 denotes in the training set otherwise 0), Gender (m/f),
      e.g., n000077, "Adhyaksa_Dault", 319, 1, m 

   -- class_overlap_vgg1_2.txt: descripe the overlapped identities (53 ids) between VGGFace1 and VGGFace2.  
   
   -- test_agetemp_imglist.txt: age templates. 
      List of pose templates for 368 identities. Each identity has 30 images. The first 10 images generate 2 front templates, 
      the second 10 ones generate 2 three-quarter templates and the third 10 ones generate 2 profile templates, 
      each template containing 5 images.
   
   -- test_posetemp_imglist.txt: pose templates.
      List of age templates for 100 identites. Each identity has 20 images. The first 10 images generate 2 young templates and 
      the second 10 images generate 2 mature templates, each template containing 5 images.
  
   -- Freebase_ID.txt: the Freebase IDs for the identities.

-- Dev_kit.tar.gz includes the demo scripts for getting tightly-cropped faces by using loosely-cropped regions on VGGFace2, and 
   GT boxes on IJB-B test set.

-- bb_landmark.tar.gz includes the information of bounding box and 5 facial landmarks estimated by MTCNN model 
   referring to the provided loosely cropped faces.


We release four models trained on the training set of VGGFace2 based on Caffe and MatConvNet implementations. 
Please refer to ReadMe.txt in the respective tar files for details. 
