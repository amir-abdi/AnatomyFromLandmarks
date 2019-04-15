# Landmarks To Shape
Generate Anatomy From Landmarks

This code accompanies the submission **"AnatomyGen: Deep Anatomy Generation From Dense Representation With Applications in Mandible Synthesis"**, 
in the Medical Imaging with Deep Learning Conference, London, 2019.
The manuscript is accessible on [openreview](https://openreview.net/pdf?id=BkltUK71xV). 

#### Citation

If you used the code or the voxelized version of the dataset in your research, 
please include the following citation:

    @inproceedings{LandmarksToShape2019,
      title={AnatomyGen: Deep Anatomy Generation From Dense Representation With Applications in Mandible Synthesis},
      author={Abdi, Amir H. and Borgard, Heather and Abolmaesumi, Purang and Fels, Sidney},
      booktitle={Medical Imaging with Deep Learning},
      series={Proceedings of Machine Learning Research (PMLR)},
      year={2019}
    }

### Data
The data that accompanies the code includes voxelized version of mandible (jaw) bones collected from the 
MICCAI 2015 segmentation challenge [1], 
Cancer Imaging Archive (TCIA) [2], 
and samples released by Wallner et al [3].
The submisison also made use of 48 more mandible samples, which are not included here, but can be accessed through 
data sharing agreements (please contact authors for further instructions).
 
 
 To download the data and set environment variables, please run 
 
     source download-data.sh
     
 You can also download the data separately [here](https://drive.google.com/uc?export=download&id=1GCslF1eo6Bz2EK207CvBfRDUReb_dyXi).
 Please set the environment variable `$MANDIBLE_DATA_PATH`  to the data directory.
 
 ### Training and Testing
This is a python3 implementation. 
To install the requirements, depending on your preferred package manager, please run:
    
    pip3 install -r requirements.txt    
    or
    conda install --file requirements.txt
  

To train the model on the training subset, run the script 

    bash train-test.sh
    
To test your model on the test subset, set the `--test` flag in the `train-test.sh` script to `true`.
     
 
 ### Licensing
This code is released under GNU GENERAL PUBLIC LICENSE V3.

 
 #### References
 [1] Patrik F. Raudaschl and  Paolo Zaffino et al. Evaluation  of  segmentation  methods  on  head  and  neck  CT:  Auto-segmentation challenge 2015.Medical Physics, 44(5):2020–2036, 2017a. \
 [2] Margarita L. Zuley and Rose Jarosz et al.  Radiology data fromthe cancer genome atlas head-neck squamous cell carcinoma collection, 2016. \
 [3] Jrgen Wallner and Jan Egger.  Mandibular ct dataset collection, 2018.
 
 