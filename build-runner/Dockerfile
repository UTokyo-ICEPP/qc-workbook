FROM ubuntu:latest

RUN ln -sf /usr/share/zoneinfo/Asia/Tokyo /etc/localtime

RUN apt-get -y update && DEBIAN_FRONTEND=noninteractive apt-get install -y lighttpd

RUN apt-get install -y python3 python3-venv python3-pip git && apt-get -y clean

RUN pip3 install --no-cache-dir qiskit jupyterlab matplotlib pylatexenc jupyter-book

RUN mkdir /build && rm -rf /var/www/html && ln -s /build/_build/html /var/www/html

EXPOSE 80

COPY build_and_publish.py /

ENTRYPOINT ["/usr/sbin/lighttpd", "-D", "-f", "/etc/lighttpd/lighttpd.conf"]
