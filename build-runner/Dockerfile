FROM ubuntu:24.04 AS packages
ADD requirements.txt /requirements.txt

RUN apt -y update && \
    apt install -y wget python3 python3-venv && \
    apt -y clean && \
    rm -rf /var/lib/apt/lists/*

RUN wget https://bootstrap.pypa.io/get-pip.py && python3 get-pip.py --prefix /usr && rm get-pip.py

RUN pip3 install --no-cache-dir --prefix /usr -r /requirements.txt && rm /requirements.txt
RUN pip3 install --no-cache-dir --prefix /usr jupyter-book

# The next two blocks are libraries required at runtime (not build time) only, so they are not needed in principle
# (Cells using these libraries are tagged as "throws-exception")
# RUN pip3 install dwave-qbsolv

# RUN apt -y install git
# RUN git clone https://github.com/LAL/trackml-library.git && \
#     pip3 install ./trackml-library && \
#     rm -rf trackml-library

FROM packages AS buildenv
ADD clean_packages.sh /clean_packages.sh

RUN /clean_packages.sh && rm /clean_packages.sh

# Build in two stages
# $ docker build -t yiiyama/jupyterbook-qiskit-packages:__version__ --target packages .
# $ docker build -t yiiyama/jupyterbook-qiskit:__version__ .
# Then push both and use the first one for the singularity container
