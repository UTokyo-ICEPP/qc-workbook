FROM ubuntu:latest AS packages
ADD requirements.txt /requirements.txt

RUN apt -y update && \
    apt install -y python3 python3-venv python3-pip && \
    apt -y clean && \
    rm -rf /var/lib/apt/lists/*

RUN python3 -m venv /opt/packages
ENV PATH="/opt/packages/bin:$PATH"

RUN pip3 install --no-cache-dir -r /requirements.txt && rm /requirements.txt
RUN pip3 install --no-cache-dir jupyter-book

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
