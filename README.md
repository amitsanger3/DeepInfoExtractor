
# Deep Info Extractor (DIE)

Deep Info Extractor (DIE) is an open-source tool that can extract useful information from micro text. Name Entity Recognition is an example of an information extraction task and DIE can extract such information but taking one task at a time, that is, tool can extract either name or entity, or location at a time. DIE tool can be trained on any labeled data that is in Conell 2003 format & provide 2 pre-trained & 6 novel pre-processing embedding methods to choose from. The tool can be used with or without docker. Docker helps you to relax from the environment variations.


## Demo

Follow this YouTube video tutorial to successfully install & run DIE docker image.

[![IMAGE ALT TEXT HERE](https://img.youtube.com/vi/R4Wa4ZNCD3M/0.jpg)](https://www.youtube.com/watch?v=R4Wa4ZNCD3M)


## Installation

#### Install DIE with Docker

```bash
  docker pull gateid/die:(latest_version)
```
*latest_version: See the [DIE docker image](https://hub.docker.com/r/gateid/die/tags) for details about latest version.

#### Normal Installation

** Install & initialize virtual environment as per you OS.
```bash
  cd DIE
  pip install -r lmr_requirements.txt 
  python api_run.py
```


    
## Environment Variables

To run this project, you will need to add the following environment variables to your .env file

`API_KEY`

`ANOTHER_API_KEY`


## Authors

- Amit Kumar Sanger
- Raghav Sharma
- Rohit Pandey
- Yatin Tomer


## Acknowledgements

 - [Docker Installation](https://docs.docker.com/engine/)
 - [BERT](https://arxiv.org/abs/1810.04805)
 - [Deberta](https://arxiv.org/abs/2006.03654)




