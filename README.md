
# Deep Info Extractor (DIE)

Deep Info Extractor (DIE) is an open-source tool that can extract useful information from micro text. Name Entity Recognition is an example of an information extraction task and DIE can extract such information but taking one task at a time, that is, tool can extract either name or entity, or location at a time. DIE tool can be trained on any labeled data that is in Conell 2003 format & provide 2 pre-trained & 6 novel pre-processing embedding methods to choose from. The tool can be used with or without docker. Docker helps you to relax from the environment variations.

## Paper

[Deep Information Extractor (DIE): A Multipurpose Information Extractor with Shifted Vectors Pre-processing](https://link.springer.com/chapter/10.1007/978-981-97-5441-0_29)

## Demo

Follow this YouTube video tutorial to successfully install & run DIE docker image.

[![IMAGE ALT TEXT HERE](https://img.youtube.com/vi/R4Wa4ZNCD3M/0.jpg)](https://www.youtube.com/watch?v=R4Wa4ZNCD3M)

NOTE: Inadvertently, the wrong pre-training weights are used in this demonstration. However, the correct weights can be used by following the same procedure. We will rectify this error in the future.

## Download Dependencies & Trained Weights

To run this project, you will need to download the DIE directory from the [google drive](https://drive.google.com/drive/folders/1I6idw9pASneTJ5BPyD0cq6nNZ3YxXQnC?usp=sharing).

To annonate your data this tutorial can help you. 

[![IMAGE ALT TEXT HERE](https://img.youtube.com/vi/CxfGJGK4mxQ/0.jpg)](https://www.youtube.com/watch?v=CxfGJGK4mxQ)

(NOTE: This is an external service, application, and tutorial. We cannot guarantee the product's quality, origin, or any other product-specific characteristics. We do not endorse any of the products, add-ons, commercials, or other business-related content or monetary benefit that the owner of this video and software shows, displays, or offers. We're directing you to this video for assistance. It is entirely up to you to utilise this video information and any products displayed, promoted, sold, or offered. We are not responsible for any damage, loss, or harm caused by the information, products, offers, or anything else displayed, promoted, sold, or offered in this video.)



#### Structure of the DIE directory

```bash
  DIE 
    |-- Pre-Trained Embedding Model
        |-- bert-base-multilingual-cased
        |-- deberta-base
        |-- pos_tagger
    |-- Trained Weights
        |-- Conll2003_Location
            |-- Conell
                |-- bert
                    |-- lmr.pt
                |-- bert_posam
                    |-- lmr.pt
                |-- deberta
                    |-- lmr.pt
                |-- deberta_posam
                    |-- lmr.pt
                |-- mdm
                    |-- lmr.pt
                |-- mdm_posam
                    |-- lmr.pt
                |-- vsm
                    |-- lmr.pt
                |-- vsm_posam
                    |-- lmr.pt
        |-- Conll2003_Organisation
            |-- Conell
                |-- bert
                    |-- lmr.pt
                |-- bert_posam
                    |-- lmr.pt
                |-- deberta
                    |-- lmr.pt
                |-- deberta_posam
                    |-- lmr.pt
                |-- mdm
                    |-- lmr.pt
                |-- mdm_posam
                    |-- lmr.pt
                |-- vsm
                    |-- lmr.pt
                |-- vsm_posam
                    |-- lmr.pt
        |-- Conll2003_Person
            |-- Conell
                |-- bert
                    |-- lmr.pt
                |-- bert_posam
                    |-- lmr.pt
                |-- deberta
                    |-- lmr.pt
                |-- deberta_posam
                    |-- lmr.pt
                |-- mdm
                    |-- lmr.pt
                |-- mdm_posam
                    |-- lmr.pt
                |-- vsm
                    |-- lmr.pt
                |-- vsm_posam
                    |-- lmr.pt

```

*In Pre-Trained Embedding Model directory you get pre-Trained SOTA weights which you need to place while you running DIE on normal environment. On Docker environment you need not to do anything, everything is already set up in the image.

**In Trained Weights directory you will get trained models on 2 SOTA embeddings & 6 DIE novel embeddings methods. You can use them for your usage.


## Installation

#### Install DIE with Docker

```bash
  docker pull gateid/die:(latest_version)
```
*latest_version: See the [DIE docker image](https://hub.docker.com/r/gateid/die/tags) for details about latest version.

```bash
  docker run -p 5000:5000 -v /shared_dir_path:/geoai die:(latest_version)
```
*shared_dir_path: complete path of the directory which you want to share with docker container. See demo video.

**5000 is default port. Its suggested not to change this for easy use.

    
## Normal Installation

Clone the project

```bash
  git clone https://github.com/amitsanger3/DeepInfoExtractor
```

Go to the project directory

```bash
  cd DeepInfoExtractor
```

Place all directories which is in 'Pre-Trained Embeddings Model' directory which you download from google drive. 

```bash
  bert-base-multilingual-cased
  deberta-base
  pos_tagger
```

Install dependencies

```bash
  pip install -r lmr_requirements.txt 
```

Open config.py file and change the 'API CONFIG' section as per the description

```bash
# ####################### API CONFIGS ######################################
# Change the  below location with the complete path of your loca dirs 
# as mention below

conell_files_dir = "/geoai/data/conll2003/"  # Path where your conll2003/dataset dir is placed
conell_train_file = "/geoai/data/conll2003/train.txt"  # Path where your conll2003/dataset train file is placed in conll2003 format
conell_valid_file = "/geoai/data/conll2003/dev.txt"  # Path where your conll2003/dataset validation file is placed in conll2003 format
conell_test_file = "/geoai/data/conll2003/test.txt"  # Path where your conll2003/dataset  test file is placed in conll2003 format

model_path = "/geoai/TRAINED_MODEL/Conell/"  # Path where you want your trained model will be saved
logs_path = "/geoai/LOGS/Conell/"  # Path where you want your logs will be saved
# ##########################################################################
```

Start the flask app

```bash
  python api_run.py
```

Open your browser and enter the below domain

```bash
  https://your_system_local_ip:5000/main
```
**e.g. : https://192.168.22.200:5000/main

VOILA !!!

The GUI app is started. Now, do the training & predictions as shown in Demo above.



## Authors

- Amit Kumar Sanger
- Raghav Sharma
- Rohit Pandey
- Yatin Tomer

## Citation

```http
  @inproceedings{sharma2024deep,
  title={Deep Information Extractor (DIE): A Multipurpose Information Extractor with Shifted Vectors Pre-processing},
  author={Sharma, Raghav and Sanger, Amit and Tomer, Yatin and Pandey, Rohit},
  booktitle={Proceedings of Ninth International Congress on Information and Communication Technology: ICICT 2024, London, Volume 10},
  pages={343},
  organization={Springer Nature}}
```

## Acknowledgements

 - [Docker Installation](https://docs.docker.com/engine/)
 - [BERT](https://arxiv.org/abs/1810.04805)
 - [Deberta](https://arxiv.org/abs/2006.03654)


## Support

For support, email amitsanger1988@gmail.com or join [Youtube channel](https://www.youtube.com/@AmitSangerdes).


