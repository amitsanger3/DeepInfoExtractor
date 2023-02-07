FROM python:3.7


RUN mkdir -p /opt/LMR

RUN mkdir -p /geoai

WORKDIR /opt/LMR

# install requirements
COPY . /opt/LMR

RUN cd /opt/LMR
RUN /bin/bash -c "source lmrGloveEnv/bin/activate"

COPY ./lmr_requirements.txt /opt/LMR/lmr_requirements.txt

RUN python -m pip install --upgrade pip

RUN pip install -r lmr_requirements.txt

#CMD ["python", "submission.py"]

#CMD ["python", "lmr_main.py", "--process_file", "True"]

CMD ["python", "api_run.py"]